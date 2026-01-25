# auto_optimize_update.py
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from loguru import logger
import gc
import warnings
from dataclasses import dataclass, field
import psutil
import multiprocessing
import math
from flask import Flask, jsonify, request
import time

warnings.filterwarnings('ignore')

# Configure logging
logger.add('/home/bumblebee/Project/optimize/log/ml_optimizer_update.log', retention='7 days', rotation="10 MB")

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = min(16, multiprocessing.cpu_count())  # Optimized for 16 core/32 thread
PIN_MEMORY = True if torch.cuda.is_available() else False

@dataclass
class MLOptimizerConfig:
    """Configuration for the ML Optimizer"""
    db_path: str = '/home/bumblebee/Project/optimize/data/lending_history.db'
    model_save_path: str = '/home/bumblebee/Project/optimize/data/ml_models_update/'
    cache_dir: str = '/home/bumblebee/Project/optimize/data/ml_cache_update/'
    periods: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 10, 14, 15, 20, 30, 60, 90, 120])
    currencies: List[str] = field(default_factory=lambda: ['fUST', 'fUSD'])
    lookback_days_list: List[int] = field(default_factory=lambda: [7, 15, 30, 60, 90])
    min_records: int = 50
    # Training Hyperparameters
    batch_size: int = 8192  # Increased for 5090 (24GB+ VRAM)
    n_epochs: int = 100
    learning_rate: float = 0.0005
    hidden_size: int = 512  # Increased model capacity
    num_layers: int = 3
    dropout: float = 0.2
    sequence_length: int = 120 # Training sequence length (e.g. 2 hours or more depending on resampling)
    prediction_horizon: int = 120 # Predict 2 hours ahead
    
    def __post_init__(self):
        # Create directories
        for dir_path in [self.model_save_path, self.cache_dir]:
            os.makedirs(dir_path, exist_ok=True)

class LSTMAttentionModel(nn.Module):
    """LSTM with Attention Mechanism and Static Features"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float, 
                 num_periods: int, num_currencies: int, embedding_dim: int = 16):
        super(LSTMAttentionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embeddings for static features
        self.period_embedding = nn.Embedding(num_periods + 1, embedding_dim)
        self.currency_embedding = nn.Embedding(num_currencies + 1, embedding_dim)
        
        # LSTM Layer
        # Input dim + embeddings
        lstm_input_dim = input_dim + (embedding_dim * 2)
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Attention Layer
        self.attention_linear = nn.Linear(hidden_dim, 1)
        
        # Output Layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x, periods, currencies):
        # x: [batch, seq_len, features]
        # periods: [batch]
        # currencies: [batch]
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Process Embeddings
        # Repeat embeddings for each time step to concatenate with time-series input
        p_emb = self.period_embedding(periods).unsqueeze(1).repeat(1, seq_len, 1) # [batch, seq_len, emb_dim]
        c_emb = self.currency_embedding(currencies).unsqueeze(1).repeat(1, seq_len, 1) # [batch, seq_len, emb_dim]
        
        # Concatenate features
        combined_input = torch.cat([x, p_emb, c_emb], dim=2)
        
        # LSTM
        lstm_out, _ = self.lstm(combined_input) # [batch, seq_len, hidden_dim]
        
        # Attention Mechanism
        # We want to weigh the importance of different time steps
        weights = torch.tanh(self.attention_linear(lstm_out)) # [batch, seq_len, 1]
        attention_weights = F.softmax(weights, dim=1) # [batch, seq_len, 1]
        
        # Context vector
        context_vector = torch.sum(attention_weights * lstm_out, dim=1) # [batch, hidden_dim]
        
        # Fully Connected
        out = self.fc1(context_vector)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        return out

class LendingDataset(Dataset):
    def __init__(self, sequences, targets, periods, currencies):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.periods = torch.LongTensor(periods)
        self.currencies = torch.LongTensor(currencies)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.periods[idx], self.currencies[idx]

class EnhancedDataLoader:
    """Optimized Data Loader using pandas and sqlite3"""
    
    def __init__(self, config: MLOptimizerConfig):
        self.config = config
        self.db_url = f'file:{self.config.db_path}?mode=ro'
        
    def get_connection(self):
        try:
            return sqlite3.connect(self.db_url, uri=True)
        except:
            return sqlite3.connect(self.config.db_path)

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load ALL required data into memory efficiently"""
        logger.info("Loading all data from database...")
        conn = self.get_connection()
        
        # Calculate max lookback needed
        max_lookback = max(self.config.lookback_days_list) + 5 # Buffer
        start_date = datetime.now() - timedelta(days=max_lookback)
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        
        query = """
        SELECT currency, period, datetime, close_annual as yield_rate, volume, hour, day_of_week, month
        FROM funding_rates 
        WHERE datetime >= ?
        ORDER BY currency, period, datetime
        """
        
        try:
            df = pd.read_sql_query(query, conn, params=(start_str,))
            conn.close()
            
            if df.empty:
                logger.warning("No data found in database.")
                return {}
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Data cleaning
            df = df[df['yield_rate'] > 0]
            df = df[df['yield_rate'] < 1000] # Reasonable upper bound
            
            # Organize by key
            data_dict = {}
            # Group by currency and period for fast access
            for (curr, per), group in df.groupby(['currency', 'period']):
                # Set index
                g = group.set_index('datetime').sort_index()
                # Resample to ensure minute-level continuity (fill missing with ffill)
                # This ensures time steps are consistent for LSTM
                # g = g.resample('1T').ffill() 
                # Note: Resampling might introduce too much synthetic data if gaps are large. 
                # For now, we assume data is mostly continuous or we accept gaps.
                
                data_dict[f"{curr}_{per}"] = g
                
            logger.info(f"Loaded {len(df)} rows across {len(data_dict)} currency/period pairs.")
            return data_dict
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            if 'conn' in locals(): conn.close()
            return {}

    def calculate_trade_metrics(self, df: pd.DataFrame, target_yield: float) -> Dict:
        """Calculate trade metrics (legacy support)"""
        if df.empty or len(df) < 10:
            return self._empty_metrics()
        
        try:
            # 1. Trade Probability
            in_target_mask = df['yield_rate'] <= target_yield
            trade_probability = float(in_target_mask.mean())
            
            # 2. Average Duration
            if in_target_mask.any():
                in_target_groups = (in_target_mask != in_target_mask.shift()).cumsum()
                durations = df[in_target_mask].groupby(in_target_groups).size()
                avg_duration_hours = float(durations.mean() / 60.0) 
            else:
                avg_duration_hours = 0.0
            
            # 3. Volatility & Sharpe
            returns = df['yield_rate'].pct_change().dropna()
            if len(returns) > 1:
                volatility = float(returns.std() * np.sqrt(365 * 24))
                avg_return = float(df['yield_rate'].mean() / 100.0)
                sharpe_ratio = float(avg_return / (volatility + 1e-6))
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
            
            # 4. Max Drawdown
            if len(returns) > 1:
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / (running_max + 1e-8)
                max_drawdown = float(drawdown.min())
            else:
                max_drawdown = 0.0
            
            return {
                'trade_probability': trade_probability,
                'avg_duration_hours': avg_duration_hours,
                'max_consecutive_hours': float(durations.max()/60.0) if in_target_mask.any() else 0.0,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_yield': float(df['yield_rate'].mean()),
                'std_yield': float(df['yield_rate'].std())
            }
        except Exception as e:
            # logger.error(f"Error metrics: {e}")
            return self._empty_metrics()

    def _empty_metrics(self):
        return {
            'trade_probability': 0.0, 'avg_duration_hours': 0.0, 'max_consecutive_hours': 0.0,
            'volatility': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
            'avg_yield': 0.0, 'std_yield': 0.0
        }

class OptimizedPredictor:
    """Main Optimization Class"""
    
    def __init__(self, config: MLOptimizerConfig):
        self.config = config
        self.data_loader = EnhancedDataLoader(config)
        self.model = None
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.period_map = {p: i for i, p in enumerate(sorted(config.periods))}
        self.currency_map = {c: i for i, c in enumerate(sorted(config.currencies))}
        self.data_cache = {}
        
    def prepare_global_dataset(self):
        """Prepares a large unified dataset for the global model"""
        self.data_cache = self.data_loader.load_all_data()
        
        all_sequences = []
        all_targets = []
        all_periods = []
        all_currencies = []
        
        feature_cols = ['yield_rate', 'volume', 'hour', 'day_of_week', 'month']
        # Pre-fit scalers on all data
        all_data_values = []
        for df in self.data_cache.values():
            all_data_values.append(df[feature_cols].values)
            
        if not all_data_values:
            return None
            
        # Global Scaling
        big_matrix = np.vstack(all_data_values)
        
        # Split into features (X) and target (yield only for y)
        # We use all features for input, but scaling must be careful
        self.scaler_X.fit(big_matrix)
        self.scaler_y.fit(big_matrix[:, 0].reshape(-1, 1)) # Yield is col 0
        
        seq_len = self.config.sequence_length
        horizon = self.config.prediction_horizon
        
        logger.info("Generating sequences...")
        
        for key, df in self.data_cache.items():
            currency, period = key.split('_')
            period = int(period)
            
            if len(df) <= seq_len + horizon:
                continue
                
            # Scale data
            data_scaled = self.scaler_X.transform(df[feature_cols].values)
            y_scaled = self.scaler_y.transform(df['yield_rate'].values.reshape(-1, 1))
            
            p_idx = self.period_map.get(period, 0)
            c_idx = self.currency_map.get(currency, 0)
            
            # Sliding window - Optimized with numpy stride tricks could be better, but loop is explicit
            # For massive data, stride_tricks is better.
            # Using a stride to reduce overlap and data size if needed, but let's take all
            
            # Limit samples per series to avoid imbalance?
            # For now, take all.
            
            # Fast sequence generation
            num_samples = len(df) - seq_len - horizon
            
            # We can vectorize this
            # X: [num_samples, seq_len, features]
            # y: [num_samples, 1] (target at t + horizon)
            
            for i in range(0, num_samples, 2): # Stride 2 to reduce redundancy slightly
                X = data_scaled[i : i+seq_len]
                y = y_scaled[i + seq_len + horizon - 1] 
                
                all_sequences.append(X)
                all_targets.append(y)
                all_periods.append(p_idx)
                all_currencies.append(c_idx)
                
        if not all_sequences:
            return None
            
        logger.info(f"Generated {len(all_sequences)} sequences.")
        
        return LendingDataset(all_sequences, all_targets, all_periods, all_currencies)

    def train_global_model(self, force=False):
        """Trains the global LSTM-Attention model"""
        model_path = os.path.join(self.config.model_save_path, 'global_model.pth')
        scaler_path = os.path.join(self.config.model_save_path, 'global_scalers.pkl')
        
        # Check if model exists and is fresh
        if not force and os.path.exists(model_path):
            try:
                # Load scalers
                with open(scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.scaler_X = scalers['scaler_X']
                    self.scaler_y = scalers['scaler_y']
                
                # Load Model
                # Need to know input dim from data or config. Assuming fixed 5 features for now.
                input_dim = 5 
                self.model = LSTMAttentionModel(
                    input_dim, self.config.hidden_size, 1, 
                    self.config.num_layers, self.config.dropout,
                    len(self.config.periods), len(self.config.currencies)
                ).to(DEVICE)
                
                self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                self.model.eval()
                logger.info("Loaded existing global model.")
                
                # Load data cache for predictions if not loaded
                if not self.data_cache:
                    self.data_cache = self.data_loader.load_all_data()
                return True
            except Exception as e:
                logger.warning(f"Failed to load model: {e}, retraining...")
        
        # Prepare Data
        dataset = self.prepare_global_dataset()
        if dataset is None:
            logger.error("Failed to prepare dataset.")
            return False
            
        input_dim = dataset[0][0].shape[1]
        
        # Train/Val Split
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, 
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, 
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        
        # Initialize Model
        self.model = LSTMAttentionModel(
            input_dim, self.config.hidden_size, 1,
            self.config.num_layers, self.config.dropout,
            len(self.config.periods), len(self.config.currencies)
        ).to(DEVICE)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        scaler = GradScaler(enabled=True) # AMP is always good for 5090
        
        logger.info(f"Starting training on {DEVICE} with {len(train_ds)} samples...")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.n_epochs):
            self.model.train()
            train_loss = 0.0
            
            for X, y, p, c in train_loader:
                X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                p, c = p.to(DEVICE, non_blocking=True), c.to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                
                with autocast(enabled=True):
                    pred = self.model(X, p, c)
                    loss = criterion(pred, y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y, p, c in val_loader:
                    X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                    p, c = p.to(DEVICE, non_blocking=True), c.to(DEVICE, non_blocking=True)
                    
                    with autocast(enabled=True):
                        pred = self.model(X, p, c)
                        loss = criterion(pred, y)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            logger.info(f"Epoch {epoch+1}/{self.config.n_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(self.model.state_dict(), model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    logger.info("Early stopping.")
                    break
        
        # Save scalers
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler_X': self.scaler_X, 'scaler_y': self.scaler_y}, f)
            
        return True

    def predict_combination(self, currency: str, period: int):
        """Predicts for a specific currency/period using variable lookback contexts"""
        key = f"{currency}_{period}"
        if key not in self.data_cache:
            return None
        
        df = self.data_cache[key]
        if len(df) < 50:
            return None
            
        p_idx = self.period_map.get(period, 0)
        c_idx = self.currency_map.get(currency, 0)
        
        feature_cols = ['yield_rate', 'volume', 'hour', 'day_of_week', 'month']
        
        results_per_lookback = []
        
        current_yield = float(df['yield_rate'].iloc[-1])
        
        for lookback_days in self.config.lookback_days_list:
            # Slice data
            lookback_cutoff = df.index[-1] - timedelta(days=lookback_days)
            sub_df = df[df.index >= lookback_cutoff]
            
            if len(sub_df) < 20:
                continue
                
            # Prepare input sequence
            # We take the LAST sequence_length points from this window
            # If window < sequence_length, we pad or take what we have?
            # Model expects variable length? LSTM can handle it, but batching usually requires padding.
            # Here we are doing single inference, so we can feed exactly what we have (up to a limit).
            
            input_data = sub_df[feature_cols].values
            # Limit to reasonable max context to prevent OOM or noise
            max_context = 2000 # e.g. last 2000 minutes
            if len(input_data) > max_context:
                input_data = input_data[-max_context:]
                
            # Scale
            input_scaled = self.scaler_X.transform(input_data)
            
            # To Tensor [1, seq_len, features]
            X_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(DEVICE)
            p_tensor = torch.LongTensor([p_idx]).to(DEVICE)
            c_tensor = torch.LongTensor([c_idx]).to(DEVICE)
            
            self.model.eval()
            with torch.no_grad():
                pred_scaled = self.model(X_tensor, p_tensor, c_tensor)
                prediction = float(self.scaler_y.inverse_transform(pred_scaled.cpu().numpy())[0][0])
            
            # Metrics
            metrics = self.data_loader.calculate_trade_metrics(sub_df, prediction)
            
            # Scoring Logic (Keeping original heuristics)
            period_factor = 1.0 / (period ** 0.3)
            risk_adjustment = max(0.5, 1.0 - abs(metrics['max_drawdown']) * 3)
            expected_return = prediction * metrics['trade_probability'] * period_factor * risk_adjustment
            
            if metrics['avg_yield'] > 0:
                stability = 1.0 - min(1.0, metrics['std_yield'] / metrics['avg_yield'])
            else:
                stability = 0.5
                
            score = expected_return * (1.0 + stability * 0.5) * (1.0 + metrics['sharpe_ratio'] / 10.0)
            
            results_per_lookback.append({
                'lookback_days': lookback_days,
                'predicted_yield': prediction,
                'current_yield': current_yield,
                'trade_probability': metrics['trade_probability'],
                'expected_return': expected_return,
                'score': score,
                'metrics': metrics,
                'period_factor': period_factor
            })
            
        if not results_per_lookback:
            return None
            
        # Aggregate logic
        weights = {7: 5, 15: 4, 30: 3, 60: 2, 90: 1}
        total_weight = sum(weights.get(r['lookback_days'], 1) for r in results_per_lookback)
        
        w_score = sum(r['score'] * weights.get(r['lookback_days'], 1) for r in results_per_lookback) / total_weight
        w_yield = sum(r['predicted_yield'] * weights.get(r['lookback_days'], 1) for r in results_per_lookback) / total_weight
        w_prob = sum(r['trade_probability'] * weights.get(r['lookback_days'], 1) for r in results_per_lookback) / total_weight
        w_ret = sum(r['expected_return'] * weights.get(r['lookback_days'], 1) for r in results_per_lookback) / total_weight
        
        # Averages
        avg_dur = np.mean([r['metrics']['avg_duration_hours'] for r in results_per_lookback])
        avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in results_per_lookback])
        avg_vol = np.mean([r['metrics']['volatility'] for r in results_per_lookback])
        avg_dd = np.mean([r['metrics']['max_drawdown'] for r in results_per_lookback])
        
        return {
            'currency': currency,
            'period': period,
            'current_yield': current_yield,
            'composite_predicted_yield': w_yield,
            'composite_trade_probability': w_prob,
            'composite_expected_return': w_ret,
            'composite_score': w_score,
            'avg_duration_hours': float(avg_dur),
            'avg_sharpe_ratio': float(avg_sharpe),
            'avg_volatility': float(avg_vol),
            'avg_max_drawdown': float(avg_dd),
            'lookback_predictions': results_per_lookback,
            'total_data_points': sum(len(df) for _ in results_per_lookback), # Approximation
            'lookback_days_used': [r['lookback_days'] for r in results_per_lookback]
        }

    def find_optimal_combination(self):
        """Finds the best lending strategy"""
        if not self.train_global_model():
            return {'status': 'error', 'message': 'Model training failed'}
            
        results = []
        for currency in self.config.currencies:
            for period in self.config.periods:
                res = self.predict_combination(currency, period)
                if res:
                    results.append(res)
                    
        if not results:
             return {'status': 'error', 'message': 'No predictions generated'}
             
        # Sort by score
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        optimal = results[0]
        
        return {
            'optimal_combination': {
                'currency': optimal['currency'],
                'period': optimal['period'],
                'rate': optimal['composite_predicted_yield'],
                'trade_probability': optimal['composite_trade_probability'],
                'expected_return': optimal['composite_expected_return'],
                'stability': 0.0 # Placeholder
            },
            'detailed_metrics': optimal,
            'top_alternatives': results[1:min(5, len(results))],
            'analysis_timestamp': datetime.now().isoformat(),
            'lookback_days_used': self.config.lookback_days_list,
            'total_combinations_evaluated': len(results),
            'hardware_utilization': self._get_hardware_info(),
            'status': 'success'
        }

    def _get_hardware_info(self):
        try:
            mem = psutil.virtual_memory()
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    'name': torch.cuda.get_device_name(0),
                    'memory_allocated_gb': torch.cuda.memory_allocated(0)/1e9,
                    'memory_total_gb': torch.cuda.get_device_properties(0).total_memory/1e9
                }
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_used_gb': mem.used/1e9,
                'gpu': gpu_info
            }
        except:
            return {}

# --- API & Main ---
app = Flask(__name__)
config = MLOptimizerConfig()
optimizer_instance = None

def get_optimizer():
    global optimizer_instance
    if optimizer_instance is None:
        optimizer_instance = OptimizedPredictor(config)
    return optimizer_instance

@app.route('/api/ml/optimize', methods=['GET'])
def optimize():
    opt = get_optimizer()
    force = request.args.get('force_retrain', 'false').lower() == 'true'
    download_history = request.args.get('download', 'false').lower() == 'true'

    if force:
        import shutil
        if os.path.exists(config.model_save_path):
            shutil.rmtree(config.model_save_path)
            os.makedirs(config.model_save_path)

    if download_history:
        # 下载最新数据
        from funding_history_downloader import main as downloader
        downloader()
        
    result = opt.find_optimal_combination()
    
    with open('data/optimal_combination_update.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
        
    return jsonify(result)

@app.route('/api/ml/optimal_combination', methods=['GET'])
def get_optimal_combination():
    """读取本地optimal_combination_update.json文件并返回结果"""
    try:
        file_path = 'data/optimal_combination_update.json'
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({
                'status': 'error',
                'message': f'File not found: {file_path}',
                'suggestions': [
                    'Run the optimization first using /api/ml/optimize',
                    'Check if the data directory exists',
                    'Ensure the program has write permissions'
                ]
            }), 404
        
        # 读取JSON文件
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 添加文件信息
        file_info = {
            'file_path': file_path,
            'file_size_kb': os.path.getsize(file_path) / 1024,
            'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            'read_timestamp': datetime.now().isoformat()
        }
        
        # 合并数据
        response_data = {
            'status': 'success',
            'file_info': file_info,
            **data
        }
        
        return jsonify(response_data)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {file_path}: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Invalid JSON format in file',
            'error': str(e)
        }), 500
        
    except Exception as e:
        logger.error(f"Error reading optimal_combination.json: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ml/simple_optimize', methods=['GET'])
def simple_optimize():
    opt = get_optimizer()
    result = opt.find_optimal_combination()
    if result['status'] == 'success':
        opt_res = result['optimal_combination']
        return jsonify({
            'status': 'success',
            'currency': opt_res['currency'],
            'period': opt_res['period'],
            'rate': opt_res['rate'],
            'trade_probability': opt_res['trade_probability'],
            'expected_return': opt_res['expected_return'],
            'timestamp': result['analysis_timestamp']
        })
    return jsonify(result)

@app.route('/api/ml/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'gpu': torch.cuda.is_available(),
        'device': str(DEVICE)
    })

def main():
    print(f"🚀 High-Performance ML Optimizer (RTX 5090 Ready)")
    print(f"   Device: {DEVICE}")
    
    import shutil
    if os.path.exists(config.model_save_path):
        shutil.rmtree(config.model_save_path)
        os.makedirs(config.model_save_path)

    from funding_history_downloader import main as downloader
    downloader()

    opt = get_optimizer()
    start = time.time()
    result = opt.find_optimal_combination()
    dur = time.time() - start
    
    if result.get('status') == 'success':
        best = result['optimal_combination']
        print(f"\n✅ Optimal: {best['currency']} for {best['period']} days")
        print(f"   Rate: {best['rate']:.4f}% | Score: {result['detailed_metrics']['composite_score']:.4f}")
        print(f"   Time: {dur:.2f}s")
        
        # Save files
        with open('/home/bumblebee/Project/optimize/data/optimal_combination_update.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
            
        simple = {
            'currency': best['currency'],
            'period': best['period'],
            'rate': best['rate'],
            'trade_probability': best['trade_probability'],
            'expected_return': best['expected_return'],
            'timestamp': result['analysis_timestamp']
        }
        with open('/home/bumblebee/Project/optimize/data/optimal_simple_update.json', 'w') as f:
            json.dump(simple, f, indent=2, default=str)
    else:
        print(f"❌ Error: {result.get('message')}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--api':
        app.run(host='0.0.0.0', port=5000)
    else:
        main()
