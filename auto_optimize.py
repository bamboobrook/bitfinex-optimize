# ml_optimizer_fixed.py
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from loguru import logger
import gc
import warnings
from dataclasses import dataclass
import psutil
import multiprocessing

warnings.filterwarnings('ignore')

# 配置日志
logger.add('log/ml_optimizer.log', retention='7 days', rotation="10 MB")

@dataclass
class MLOptimizerConfig:
    """机器学习优化器配置"""
    db_path: str = 'data/lending_history.db'
    model_save_path: str = 'data/ml_models/'
    cache_dir: str = 'data/ml_cache/'
    periods: List[int] = None
    currencies: List[str] = None
    lookback_days_list: List[int] = None
    min_records: int = 50  # 进一步降低最小记录数要求
    batch_size: int = 4096
    n_epochs: int = 200
    learning_rate: float = 0.001
    use_gpu: bool = True
    use_amp: bool = False  # 暂时禁用AMP，减少复杂性
    
    def __post_init__(self):
        if self.periods is None:
            self.periods = [2, 3, 4, 5, 6, 7, 10, 14, 15, 20, 30, 60, 90, 120]
        if self.currencies is None:
            self.currencies = ['fUST', 'fUSD']
        if self.lookback_days_list is None:
            self.lookback_days_list = [7, 15, 30, 60, 90]
        
        # 创建目录
        for dir_path in [self.model_save_path, self.cache_dir]:
            os.makedirs(dir_path, exist_ok=True)


class EnhancedDataLoader:
    """增强数据加载器"""
    
    def __init__(self, config: MLOptimizerConfig):
        self.config = config
        self.conn = None
        
    def connect(self):
        """连接到数据库"""
        try:
            self.conn = sqlite3.connect(f'file:{self.config.db_path}?mode=ro', uri=True)
            logger.info(f"Connected to database: {self.config.db_path}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            # 尝试常规连接
            try:
                self.conn = sqlite3.connect(self.config.db_path)
                logger.info(f"Connected to database (read-write mode): {self.config.db_path}")
                return True
            except Exception as e2:
                logger.error(f"Failed to connect in any mode: {e2}")
                return False
    
    def disconnect(self):
        """断开数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def get_recent_data(self, currency: str, period: int, lookback_days: int) -> pd.DataFrame:
        """获取最近的分钟级数据"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            
            query = """
            SELECT 
                datetime,
                close_annual as yield_rate,
                volume,
                hour,
                day_of_week,
                month
            FROM funding_rates 
            WHERE currency = ? AND period = ? AND datetime >= ?
            ORDER BY datetime
            """
            
            df = pd.read_sql_query(query, self.conn, params=(currency, period, start_str))
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)
                
                # 数据质量控制
                df = df[df['yield_rate'] > 0]  # 移除负利率
                df = df[df['yield_rate'] < 50]  # 移除异常高利率
                
                logger.info(f"Loaded {len(df)} records for {currency}_{period}_{lookback_days}d")
                return df
            else:
                logger.warning(f"No data found for {currency}_{period}_{lookback_days}d")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching data for {currency}_{period}_{lookback_days}d: {e}")
            return pd.DataFrame()
    
    def calculate_trade_metrics(self, df: pd.DataFrame, target_yield: float) -> Dict:
        """计算交易指标"""
        if df.empty or len(df) < 10:
            return {
                'trade_probability': 0.0,
                'avg_duration_hours': 0.0,
                'max_consecutive_hours': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_yield': 0.0,
                'std_yield': 0.0
            }
        
        try:
            # 1. 成交概率
            in_target_mask = df['yield_rate'] <= target_yield
            trade_probability = float(in_target_mask.mean())
            
            # 2. 平均持续时间
            if in_target_mask.any():
                in_target_groups = (in_target_mask != in_target_mask.shift()).cumsum()
                durations = df[in_target_mask].groupby(in_target_groups).size()
                avg_duration_hours = float(durations.mean() / 60.0) if not durations.empty else 0.0
                max_duration_hours = float(durations.max() / 60.0) if not durations.empty else 0.0
            else:
                avg_duration_hours = 0.0
                max_duration_hours = 0.0
            
            # 3. 波动率和夏普比率
            returns = df['yield_rate'].pct_change().dropna()
            if len(returns) > 1:
                volatility = float(returns.std() * np.sqrt(365 * 24))
                avg_return = float(df['yield_rate'].mean() / 100.0)
                sharpe_ratio = float(avg_return / (volatility + 1e-6))
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
            
            # 4. 最大回撤
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
                'max_consecutive_hours': max_duration_hours,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_yield': float(df['yield_rate'].mean()),
                'std_yield': float(df['yield_rate'].std())
            }
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {
                'trade_probability': 0.0,
                'avg_duration_hours': 0.0,
                'max_consecutive_hours': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_yield': 0.0,
                'std_yield': 0.0
            }


class ImprovedPredictor(nn.Module):
    """改进的预测模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 256):
        super(ImprovedPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward(self, x):
        return self.network(x)


class OptimizedGPUOptimizer:
    """优化的GPU优化器"""
    
    def __init__(self, config: MLOptimizerConfig):
        self.config = config
        self.data_loader = EnhancedDataLoader(config)
        self.device = self._setup_device()
        self.models = {}
        self.scalers = {}
        
        logger.info(f"Using device: {self.device}")
    
    def _setup_device(self):
        """设置计算设备"""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            
            # 设置CUDA优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # 设置环境变量
            os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU available: {gpu_name}")
            logger.info(f"GPU memory: {gpu_memory:.2f} GB")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for computation")
        
        return device
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备特征"""
        if df.empty or len(df) < self.config.min_records:
            return np.array([]), np.array([]), []
        
        features = pd.DataFrame(index=df.index)
        
        # 基础特征
        features['yield'] = df['yield_rate']
        features['volume'] = df['volume']
        
        # 时间特征
        features['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 简单的技术指标
        windows = [5, 10, 20, 30]
        for window in windows:
            features[f'ma_{window}'] = df['yield_rate'].rolling(window, min_periods=1).mean()
            features[f'std_{window}'] = df['yield_rate'].rolling(window, min_periods=1).std().fillna(0)
        
        # 成交量特征
        features['volume_ma_5'] = df['volume'].rolling(5, min_periods=1).mean()
        features['volume_ma_20'] = df['volume'].rolling(20, min_periods=1).mean()
        
        # 目标：未来2小时（120分钟）的收益率
        # 注意：如果数据频率不是每分钟，需要调整
        future_minutes = 120
        features['target'] = df['yield_rate'].shift(-future_minutes).fillna(method='ffill')
        
        # 删除NaN值
        features = features.dropna()
        
        if len(features) < self.config.min_records:
            logger.warning(f"Features insufficient after dropping NaN: {len(features)} records")
            return np.array([]), np.array([]), []
        
        # 分离特征和目标
        feature_cols = [col for col in features.columns if col != 'target']
        X = features[feature_cols].values
        y = features['target'].values.reshape(-1, 1)
        
        logger.info(f"Prepared features: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_cols
    
    def train_single_model(self, currency: str, period: int, lookback_days: int) -> Optional[Dict]:
        """训练单个模型"""
        key = f"{currency}_{period}_{lookback_days}"
        
        # 获取数据
        df = self.data_loader.get_recent_data(currency, period, lookback_days)
        
        if df.empty or len(df) < self.config.min_records:
            logger.warning(f"Skipping {key}: insufficient data ({len(df)} records)")
            return None
        
        logger.info(f"Training model for {key} with {len(df)} records")
        
        # 准备特征
        X, y, feature_cols = self.prepare_features(df)
        
        if len(X) == 0 or len(y) == 0:
            logger.warning(f"No features extracted for {key}")
            return None
        
        # 数据分割
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test_scaled)
        
        # 创建模型
        input_size = X_train.shape[1]
        model = ImprovedPredictor(input_size=input_size, hidden_size=256).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # 数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=min(self.config.batch_size, len(train_dataset)),
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # 训练循环
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 20
        
        for epoch in range(self.config.n_epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证
            model.eval()
            with torch.no_grad():
                X_test_gpu = X_test_tensor.to(self.device, non_blocking=True)
                y_test_gpu = y_test_tensor.to(self.device, non_blocking=True)
                
                val_predictions = model(X_test_gpu)
                val_loss = criterion(val_predictions, y_test_gpu)
                
                # 反标准化以获得实际值
                val_predictions_cpu = val_predictions.cpu().numpy()
                y_test_cpu = y_test_tensor.numpy()
                
                val_predictions_actual = scaler_y.inverse_transform(val_predictions_cpu)
                y_test_actual = scaler_y.inverse_transform(y_test_cpu)
                
                # 计算指标
                mae = float(mean_absolute_error(y_test_actual, val_predictions_actual))
                rmse = float(np.sqrt(mean_squared_error(y_test_actual, val_predictions_actual)))
                r2 = float(r2_score(y_test_actual, val_predictions_actual))
            
            if val_loss.item() < best_loss:
                best_loss = float(val_loss.item())
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 20 == 0 or epoch == self.config.n_epochs - 1:
                logger.info(f"  Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
                          f"Val Loss={val_loss.item():.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # 保存模型
        if best_model_state is not None:
            model_save_path = f"{self.config.model_save_path}{key}_model.pth"
            scaler_save_path = f"{self.config.model_save_path}{key}_scalers.pkl"
            
            torch.save(best_model_state, model_save_path)
            
            with open(scaler_save_path, 'wb') as f:
                pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
            
            # 缓存模型和标准化器
            model.load_state_dict(best_model_state)
            self.models[key] = model
            self.scalers[key] = {'scaler_X': scaler_X, 'scaler_y': scaler_y}
            
            # 清理内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            return {
                'model_path': model_save_path,
                'scaler_path': scaler_save_path,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'best_val_loss': best_loss
                },
                'input_size': input_size,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'lookback_days': lookback_days
            }
        
        return None
    
    def load_or_train_model(self, currency: str, period: int, lookback_days: int) -> bool:
        """加载或训练模型"""
        key = f"{currency}_{period}_{lookback_days}"
        
        # 检查是否已缓存
        if key in self.models and key in self.scalers:
            return True
        
        # 检查是否已有保存的模型文件
        model_path = f"{self.config.model_save_path}{key}_model.pth"
        scaler_path = f"{self.config.model_save_path}{key}_scalers.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                # 加载标准化器
                with open(scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
                
                # 获取数据以确定输入大小
                df = self.data_loader.get_recent_data(currency, period, lookback_days)
                if df.empty:
                    return False
                
                X, _, feature_cols = self.prepare_features(df)
                if len(X) == 0:
                    return False
                
                input_size = X.shape[1]
                
                # 加载模型
                model = ImprovedPredictor(input_size=input_size, hidden_size=256).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                
                # 缓存
                self.models[key] = model
                self.scalers[key] = scalers
                
                logger.info(f"Loaded model for {key}")
                return True
            except Exception as e:
                logger.error(f"Error loading model for {key}: {e}")
                # 加载失败，尝试重新训练
                result = self.train_single_model(currency, period, lookback_days)
                return result is not None
        
        # 训练新模型
        result = self.train_single_model(currency, period, lookback_days)
        return result is not None
    
    def predict_for_combination(self, currency: str, period: int, lookback_days: int) -> Optional[Dict]:
        """为特定组合和回测天数进行预测"""
        key = f"{currency}_{period}_{lookback_days}"
        
        logger.info(f"Predicting for {key}")
        
        # 确保模型已加载或训练
        if not self.load_or_train_model(currency, period, lookback_days):
            logger.warning(f"Failed to load/train model for {key}")
            return None
        
        # 获取最新数据
        df = self.data_loader.get_recent_data(currency, period, lookback_days)
        
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data for prediction: {len(df)} records")
            return None
        
        try:
            model = self.models[key]
            scalers = self.scalers[key]
            
            # 准备特征
            X, _, feature_cols = self.prepare_features(df)
            
            if len(X) == 0:
                logger.warning(f"No features for prediction")
                return None
            
            # 使用最近的数据点
            X_recent = X[-1:].copy()
            
            # 标准化
            X_scaled = scalers['scaler_X'].transform(X_recent)
            
            # 预测
            X_tensor = torch.FloatTensor(X_scaled).to(self.device, non_blocking=True)
            
            model.eval()
            with torch.no_grad():
                prediction_scaled = model(X_tensor)
                prediction = float(scalers['scaler_y'].inverse_transform(
                    prediction_scaled.cpu().numpy()
                )[0][0])
            
            # 获取当前收益率
            current_yield = float(df['yield_rate'].iloc[-1])
            
            # 计算交易指标
            trade_metrics = self.data_loader.calculate_trade_metrics(df, prediction)
            
            # 计算预期收益率
            period_factor = 1.0 / (period ** 0.3)  # 周期越短，因子越高
            risk_adjustment = max(0.5, 1.0 - abs(trade_metrics['max_drawdown']) * 3)
            
            expected_return = float(
                prediction * 
                trade_metrics['trade_probability'] * 
                period_factor * 
                risk_adjustment
            )
            
            # 计算综合得分
            if trade_metrics['avg_yield'] > 0:
                stability_score = 1.0 - min(1.0, trade_metrics['std_yield'] / trade_metrics['avg_yield'])
            else:
                stability_score = 0.5
            
            stability_score = max(0, min(1, stability_score))
            
            score = float(
                expected_return * 
                (1.0 + stability_score * 0.5) * 
                (1.0 + trade_metrics['sharpe_ratio'] / 10.0)
            )
            
            return {
                'currency': currency,
                'period': period,
                'lookback_days': lookback_days,
                'current_yield': current_yield,
                'predicted_yield': prediction,
                'trade_probability': trade_metrics['trade_probability'],
                'expected_return': expected_return,
                'avg_duration_hours': trade_metrics['avg_duration_hours'],
                'sharpe_ratio': trade_metrics['sharpe_ratio'],
                'volatility': trade_metrics['volatility'],
                'max_drawdown': trade_metrics['max_drawdown'],
                'stability_score': stability_score,
                'period_factor': period_factor,
                'score': score,
                'data_points': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {key}: {e}")
            return None
    
    def predict_all_lookback_periods(self, currency: str, period: int) -> Dict:
        """预测所有回测周期并计算综合得分"""
        logger.info(f"Predicting all lookback periods for {currency}_{period}")
        
        all_predictions = []
        failed_lookbacks = []
        
        for lookback_days in self.config.lookback_days_list:
            result = self.predict_for_combination(currency, period, lookback_days)
            if result:
                all_predictions.append(result)
                logger.info(f"  {lookback_days}d: predicted {result['predicted_yield']:.4f}% (score: {result['score']:.4f})")
            else:
                failed_lookbacks.append(lookback_days)
                logger.warning(f"  {lookback_days}d: prediction failed")
        
        # 如果有失败的，尝试使用统计方法
        for lookback_days in failed_lookbacks:
            result = self.statistical_fallback(currency, period, lookback_days)
            if result:
                all_predictions.append(result)
                logger.info(f"  {lookback_days}d (statistical): predicted {result['predicted_yield']:.4f}%")
        
        if not all_predictions:
            logger.error(f"No predictions for {currency}_{period}")
            return None
        
        # 计算加权综合得分
        weights = {7: 5, 15: 4, 30: 3, 60: 2, 90: 1}
        total_weight = sum(weights.get(p['lookback_days'], 1) for p in all_predictions)
        
        # 加权平均各指标
        weighted_scores = []
        weighted_predicted_yield = 0
        weighted_expected_return = 0
        weighted_trade_probability = 0
        
        for pred in all_predictions:
            weight = weights.get(pred['lookback_days'], 1)
            weighted_scores.append(pred['score'] * weight)
            weighted_predicted_yield += pred['predicted_yield'] * weight
            weighted_expected_return += pred['expected_return'] * weight
            weighted_trade_probability += pred['trade_probability'] * weight
        
        # 计算综合得分和指标
        composite_score = float(sum(weighted_scores) / total_weight)
        composite_predicted_yield = float(weighted_predicted_yield / total_weight)
        composite_expected_return = float(weighted_expected_return / total_weight)
        composite_trade_probability = float(weighted_trade_probability / total_weight)
        
        # 获取当前收益率
        current_yield = all_predictions[-1]['current_yield']
        
        # 获取其他指标的平均值
        avg_duration = float(np.mean([p['avg_duration_hours'] for p in all_predictions]))
        avg_sharpe = float(np.mean([p['sharpe_ratio'] for p in all_predictions]))
        avg_volatility = float(np.mean([p['volatility'] for p in all_predictions]))
        avg_max_drawdown = float(np.mean([p['max_drawdown'] for p in all_predictions]))
        
        logger.info(f"Composite score for {currency}_{period}: {composite_score:.4f}")
        
        return {
            'currency': currency,
            'period': period,
            'current_yield': current_yield,
            'composite_predicted_yield': composite_predicted_yield,
            'composite_trade_probability': composite_trade_probability,
            'composite_expected_return': composite_expected_return,
            'composite_score': composite_score,
            'avg_duration_hours': avg_duration,
            'avg_sharpe_ratio': avg_sharpe,
            'avg_volatility': avg_volatility,
            'avg_max_drawdown': avg_max_drawdown,
            'lookback_predictions': all_predictions,
            'total_data_points': int(sum([p['data_points'] for p in all_predictions])),
            'lookback_days_used': [p['lookback_days'] for p in all_predictions],
            'period_factor': all_predictions[0].get('period_factor', 1.0) if all_predictions else 1.0
        }
    
    def statistical_fallback(self, currency: str, period: int, lookback_days: int) -> Optional[Dict]:
        """统计方法后备方案"""
        logger.info(f"Using statistical fallback for {currency}_{period}_{lookback_days}d")
        
        df = self.data_loader.get_recent_data(currency, period, lookback_days)
        
        if df.empty or len(df) < 50:
            return None
        
        yields = df['yield_rate'].values
        
        # 使用多个分位数进行预测
        quantiles = [0.4, 0.5, 0.6, 0.7]
        best_score = -float('inf')
        best_result = None
        
        for q in quantiles:
            predicted_yield = float(np.percentile(yields, q * 100))
            trade_probability = float(np.mean(yields <= predicted_yield))
            
            # 计算交易指标
            trade_metrics = self.data_loader.calculate_trade_metrics(df, predicted_yield)
            
            # 计算预期收益
            period_factor = 1.0 / (period ** 0.3)
            risk_adjustment = max(0.5, 1.0 - abs(trade_metrics['max_drawdown']) * 3)
            expected_return = predicted_yield * trade_probability * period_factor * risk_adjustment
            
            # 稳定性评分
            if trade_metrics['avg_yield'] > 0:
                stability_score = 1.0 - min(1.0, trade_metrics['std_yield'] / trade_metrics['avg_yield'])
            else:
                stability_score = 0.5
            
            # 综合得分
            score = expected_return * (1.0 + stability_score * 0.5)
            
            if score > best_score:
                best_score = score
                best_result = {
                    'predicted_yield': predicted_yield,
                    'trade_probability': trade_probability,
                    'expected_return': expected_return,
                    'score': score,
                    'trade_metrics': trade_metrics,
                    'stability_score': stability_score,
                    'quantile': q
                }
        
        if best_result:
            return {
                'currency': currency,
                'period': period,
                'lookback_days': lookback_days,
                'current_yield': float(df['yield_rate'].iloc[-1]),
                'predicted_yield': float(best_result['predicted_yield']),
                'trade_probability': float(best_result['trade_probability']),
                'expected_return': float(best_result['expected_return']),
                'avg_duration_hours': best_result['trade_metrics']['avg_duration_hours'],
                'sharpe_ratio': best_result['trade_metrics']['sharpe_ratio'],
                'volatility': best_result['trade_metrics']['volatility'],
                'max_drawdown': best_result['trade_metrics']['max_drawdown'],
                'stability_score': best_result['stability_score'],
                'score': float(best_result['score']),
                'data_points': len(df),
                'method': 'statistical',
                'period_factor': 1.0 / (period ** 0.3)
            }
        
        return None
    
    def find_optimal_combination(self) -> Dict:
        """寻找最优组合"""
        logger.info("Finding optimal combination...")
        
        if not self.data_loader.connect():
            return {'status': 'error', 'message': 'Database connection failed'}
        
        try:
            all_results = []
            
            for currency in self.config.currencies:
                for period in self.config.periods:
                    logger.info(f"Processing {currency} period={period}")
                    
                    result = self.predict_all_lookback_periods(currency, period)
                    if result:
                        all_results.append(result)
                    else:
                        logger.warning(f"No results for {currency}_{period}")
            
            if not all_results:
                logger.error("No valid predictions found for any combination")
                return {'status': 'error', 'message': 'No valid predictions found'}
            
            # 按综合得分排序
            all_results.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # 获取最优组合
            optimal = all_results[0]
            
            # 生成报告
            report = {
                'optimal_combination': {
                    'currency': optimal['currency'],
                    'period': optimal['period'],
                    'rate': optimal['composite_predicted_yield'],
                    'trade_probability': optimal['composite_trade_probability'],
                    'expected_return': optimal['composite_expected_return'],
                    'stability': optimal.get('composite_stability', 0.5)
                },
                'detailed_metrics': optimal,
                'top_alternatives': all_results[1:min(5, len(all_results))],
                'analysis_timestamp': datetime.now().isoformat(),
                'lookback_days_used': self.config.lookback_days_list,
                'total_combinations_evaluated': len(all_results),
                'hardware_utilization': self._get_hardware_utilization(),
                'status': 'success'
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error finding optimal combination: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
        finally:
            self.data_loader.disconnect()
    
    def _get_hardware_utilization(self) -> Dict:
        """获取硬件利用率"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            gpu_memory_used = 0
            gpu_memory_total = 0
            
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            return {
                'cpu_usage_percent': cpu_percent,
                'memory_used_gb': memory.used / 1e9,
                'memory_total_gb': memory.total / 1e9,
                'memory_percent': memory.percent,
                'gpu_memory_used_gb': gpu_memory_used,
                'gpu_memory_total_gb': gpu_memory_total,
                'gpu_memory_percent': (gpu_memory_used / gpu_memory_total * 100) if gpu_memory_total > 0 else 0
            }
        except:
            return {}


# Flask API集成
from flask import Flask, jsonify, request

app = Flask(__name__)

# 全局配置
config = MLOptimizerConfig(
    db_path='data/lending_history.db',
    model_save_path='data/ml_models_advanced/',
    cache_dir='data/ml_cache_advanced/',
    periods=[2, 3, 4, 5, 6, 7, 10, 14, 15, 20, 30, 60, 90, 120],
    currencies=['fUST', 'fUSD'],
    lookback_days_list=[7, 15, 30, 60, 90],
    min_records=20,
    batch_size=4096,
    n_epochs=200,
    learning_rate=0.001,
    use_gpu=True,
    use_amp=False
)

# 创建优化器实例
optimizer = None

def create_optimizer():
    """创建优化器实例"""
    global optimizer
    if optimizer is None:
        optimizer = OptimizedGPUOptimizer(config)
    return optimizer

@app.route('/api/ml/optimize', methods=['GET'])
def optimize():
    """机器学习优化接口"""
    try:
        optimizer_instance = create_optimizer()
        
        force_retrain = request.args.get('force_retrain', 'false').lower() == 'true'
        download_history = request.args.get('download', 'false').lower() == 'true'
        
        if force_retrain:
            # 清理模型缓存
            optimizer_instance.models.clear()
            optimizer_instance.scalers.clear()
            
            # 删除模型文件
            import shutil
            if os.path.exists(config.model_save_path):
                shutil.rmtree(config.model_save_path)
                os.makedirs(config.model_save_path, exist_ok=True)
                
        if download_history:
            # 下载最新数据
            from funding_history_downloader import main as downloader
            downloader()
        
        # 运行优化
        result = optimizer_instance.find_optimal_combination()
        
        # 保存结果
        output_file = 'data/optimal_combination.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n📄 Full result saved to: {output_file}")
        
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ml/health', methods=['GET'])
def health():
    """健康检查接口"""
    try:
        optimizer_instance = create_optimizer()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'database_exists': os.path.exists(config.db_path),
            'config': {
                'currencies': config.currencies,
                'periods': config.periods,
                'lookback_days': config.lookback_days_list,
                'min_records': config.min_records
            },
            'hardware': {
                'cpu_cores': multiprocessing.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / 1e9
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ml/simple_optimize', methods=['GET'])
def simple_optimize():
    """简化优化接口 - 只返回最优组合"""
    try:
        optimizer_instance = create_optimizer()
        
        result = optimizer_instance.find_optimal_combination()
        
        if result.get('status') == 'success':
            optimal = result['optimal_combination']
            return jsonify({
                'status': 'success',
                'currency': optimal['currency'],
                'period': optimal['period'],
                'rate': optimal['rate'],
                'trade_probability': optimal['trade_probability'],
                'expected_return': optimal['expected_return'],
                'timestamp': result['analysis_timestamp']
            })
        else:
            return jsonify(result)
    except Exception as e:
        logger.error(f"Simple optimize error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
@app.route('/api/ml/optimal_combination', methods=['GET'])
def get_optimal_combination():
    """读取本地optimal_combination.json文件并返回结果"""
    try:
        file_path = 'data/optimal_combination.json'
        
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

def main():
    """主函数"""
    print("=" * 80)
    print("🚀 OPTIMIZED ML OPTIMIZER FOR BITFINEX LENDING")
    print("=" * 80)
    
    # 检查数据库
    if not os.path.exists(config.db_path):
        print(f"❌ Database not found: {config.db_path}")
        print("Please run the downloader first to collect data.")
        return
    
    print(f"✅ Database found: {config.db_path}")
    
    # 检查文件大小
    try:
        db_size = os.path.getsize(config.db_path) / (1024*1024)
        print(f"   Database size: {db_size:.2f} MB")
    except:
        pass
    
    # 检查硬件
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / 1e9
    
    print(f"\n💻 HARDWARE INFORMATION:")
    print(f"   CPU Cores: {cpu_count}")
    print(f"   Total Memory: {memory_gb:.1f} GB")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.2f} GB")
    else:
        print("⚠️  No GPU detected, using CPU")
    
    print(f"\n⚙️  CONFIGURATION:")
    print(f"   Currencies: {config.currencies}")
    print(f"   Periods: {config.periods}")
    print(f"   Lookback days: {config.lookback_days_list}")
    print(f"   Minimum records: {config.min_records}")
    print(f"   Batch size: {config.batch_size}")
    
    # 创建优化器
    global optimizer
    optimizer = OptimizedGPUOptimizer(config)
    
    print("\n🚀 STARTING OPTIMIZATION...")
    start_time = datetime.now()
    
    # 运行优化
    result = optimizer.find_optimal_combination()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if result.get('status') == 'success':
        optimal = result['optimal_combination']
        detailed = result['detailed_metrics']
        
        print("\n" + "=" * 80)
        print("🎯 OPTIMAL COMBINATION FOUND")
        print("=" * 80)
        print(f"Currency: {optimal['currency']}")
        print(f"Period: {optimal['period']} days")
        print(f"Predicted Rate: {optimal['rate']:.4f}%")
        print(f"Current Rate: {detailed['current_yield']:.4f}%")
        print(f"Trade Probability: {optimal['trade_probability']:.2%}")
        print(f"Expected Return: {optimal['expected_return']:.4f}")
        print()
        print("📈 DETAILED METRICS:")
        print(f"   Composite Score: {detailed['composite_score']:.4f}")
        print(f"   Avg Sharpe Ratio: {detailed['avg_sharpe_ratio']:.3f}")
        print(f"   Avg Volatility: {detailed['avg_volatility']:.4f}")
        print(f"   Avg Max Drawdown: {detailed['avg_max_drawdown']:.2%}")
        print(f"   Avg Duration: {detailed['avg_duration_hours']:.1f} hours")
        print(f"   Total Data Points: {detailed['total_data_points']:,}")
        print(f"   Lookback Periods Used: {detailed['lookback_days_used']}")
        print()
        print("⚡ PERFORMANCE:")
        print(f"   Processing Time: {duration:.2f} seconds")
        print(f"   Combinations Evaluated: {result['total_combinations_evaluated']}")
        print("=" * 80)
        
        # 保存结果
        output_file = 'data/optimal_combination.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n📄 Full result saved to: {output_file}")
        
        # 保存简化的最优组合
        simple_output = {
            'currency': optimal['currency'],
            'period': optimal['period'],
            'rate': optimal['rate'],
            'trade_probability': optimal['trade_probability'],
            'expected_return': optimal['expected_return'],
            'timestamp': result['analysis_timestamp'],
            'composite_score': detailed['composite_score']
        }
        
        with open('data/optimal_simple.json', 'w') as f:
            json.dump(simple_output, f, indent=2, default=str)
        
        print(f"📄 Simplified result saved to: data/optimal_simple.json")
        
    else:
        print(f"\n❌ Optimization failed: {result.get('message', 'Unknown error')}")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Check if database has data")
        print("   2. Check database schema matches expected format")
        print("   3. Try lowering min_records in config")
        print("   4. Check log file for details: log/ml_optimizer.log")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--api':
        # 启动Flask API
        print("Starting ML Optimizer API on http://localhost:5000")
        print("Available endpoints:")
        print("  GET /api/ml/optimize - Get optimal combination (full version)")
        print("  GET /api/ml/simple_optimize - Get optimal combination (simple version)")
        print("  GET /api/ml/health - Health check")
        print("\nPress Ctrl+C to stop")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    elif len(sys.argv) > 1 and sys.argv[1] == '--simple':
        # 只运行简单优化
        import requests
        response = requests.get('http://localhost:5000/api/ml/simple_optimize')
        print(json.dumps(response.json(), indent=2))
    else:
        # 直接运行优化
        try:
            main()
        except KeyboardInterrupt:
            print("\n\n⚠️  Optimization interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            print(f"\n❌ Fatal error: {e}")
