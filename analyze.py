# ml_optimizer_fixed_v2.py
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from loguru import logger
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
import gc
import warnings
import hashlib
warnings.filterwarnings('ignore')

# 配置日志
logger.add('log/ml_optimizer.log', retention='7 days')


class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，支持numpy数据类型"""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8, np.uint8)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return super().default(obj)


@dataclass
class MLOptimizerConfig:
    """机器学习优化器配置"""
    db_path: str = 'data/lending_history.db'
    model_save_path: str = 'data/ml_models/'
    cache_dir: str = 'data/ml_cache/'
    periods: List[int] = None
    currencies: List[str] = None
    lookback_days_list: List[int] = None  # 改为多个回测周期
    min_records: int = 300   # 降低最小记录数要求
    batch_size: int = 4096
    n_epochs: int = 500      # 进一步减少训练轮数
    learning_rate: float = 0.001
    use_gpu: bool = True
    
    def __post_init__(self):
        if self.periods is None:
            self.periods = [2, 3, 4, 5, 6, 7, 10, 14, 15, 20, 30, 60, 90, 120]
        if self.currencies is None:
            self.currencies = ['fUST', 'fUSD']
        if self.lookback_days_list is None:
            self.lookback_days_list = [7, 15, 30, 60]  # 多个回测周期
        
        # 创建目录
        for dir_path in [self.model_save_path, self.cache_dir]:
            os.makedirs(dir_path, exist_ok=True)


class EfficientDataLoader:
    """高效数据加载器"""
    
    def __init__(self, config: MLOptimizerConfig):
        self.config = config
        self.conn = None
        
    def connect(self):
        """连接到数据库"""
        try:
            self.conn = sqlite3.connect(self.config.db_path)
            logger.info(f"Connected to database: {config.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """断开数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def get_recent_data(self, currency: str, period: int, lookback_days: int) -> pd.DataFrame:
        """获取最近的分钟级数据"""
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
        
        try:
            df = pd.read_sql_query(query, self.conn, params=(currency, period, start_str))
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)
                
                # 数据质量控制
                df = df[df['yield_rate'] > 0]  # 移除负利率
                df = df[df['yield_rate'] < 100]  # 移除异常高利率
                
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {currency} period={period} days={lookback_days}: {e}")
            return pd.DataFrame()
    
    def calculate_trade_metrics(self, df: pd.DataFrame, target_yield: float) -> Dict:
        """计算交易指标"""
        if df.empty:
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
            drawdown = (cumulative - running_max) / running_max
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


class FastPredictor(nn.Module):
    """快速预测模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super(FastPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        return self.network(x)


class SimpleGPUOptimizer:
    """简单GPU优化器"""
    
    def __init__(self, config: MLOptimizerConfig):
        self.config = config
        self.data_loader = EfficientDataLoader(config)
        self.device = self._setup_device()
        self.models = {}
        self.scalers = {}
        
        logger.info(f"Using device: {self.device}")
    
    def _setup_device(self):
        """设置计算设备"""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU available: {gpu_name}")
            logger.info(f"GPU memory: {gpu_memory:.2f} GB")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for computation")
        
        return device
    
    def prepare_features_simple(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备简化特征"""
        if df.empty or len(df) < 100:
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
        
        # 简单的技术指标
        features['ma_5'] = df['yield_rate'].rolling(5, min_periods=1).mean()
        features['ma_30'] = df['yield_rate'].rolling(30, min_periods=1).mean()
        features['std_30'] = df['yield_rate'].rolling(30, min_periods=1).std().fillna(0)
        
        # 修复FutureWarning
        features['momentum_5'] = df['yield_rate'] - df['yield_rate'].shift(5).bfill().fillna(0)
        features['volume_change'] = df['volume'].pct_change().fillna(0)
        
        # 目标：未来5分钟的收益率
        features['target'] = df['yield_rate'].shift(-5).bfill().fillna(method='ffill')
        
        # 删除NaN值
        features = features.dropna()
        
        if len(features) < 100:
            return np.array([]), np.array([]), []
        
        # 分离特征和目标
        feature_cols = [col for col in features.columns if col != 'target']
        X = features[feature_cols].values
        y = features['target'].values.reshape(-1, 1)
        
        return X, y, feature_cols
    
    def train_single_model(self, currency: str, period: int, lookback_days: int) -> Optional[Dict]:
        """训练单个模型"""
        key = f"{currency}_{period}_{lookback_days}"
        
        # 获取数据
        df = self.data_loader.get_recent_data(currency, period, lookback_days)
        
        if df.empty or len(df) < self.config.min_records:
            logger.info(f"Skipping {key}: insufficient data ({len(df)} records)")
            return None
        
        logger.info(f"Training model for {key} with {len(df)} records")
        
        # 准备特征
        X, y, feature_cols = self.prepare_features_simple(df)
        
        if len(X) == 0 or len(y) == 0:
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
        
        # 转换为张量（保持在CPU上）
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test_scaled)
        
        # 创建模型
        input_size = X_train.shape[1]
        model = FastPredictor(input_size=input_size, hidden_size=128).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # 数据加载器（不使用pin_memory，因为我们在循环中手动转移到GPU）
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=min(self.config.batch_size, len(train_dataset)), 
                                  shuffle=True)
        
        # 训练循环
        best_loss = float('inf')
        best_model_state = None
        
        for epoch in range(self.config.n_epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # 手动将batch转移到GPU
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_test_tensor.to(self.device))
                val_loss = criterion(val_predictions, y_test_tensor.to(self.device))
                
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
            
            if epoch % 10 == 0 or epoch == self.config.n_epochs - 1:
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
                
                X, _, feature_cols = self.prepare_features_simple(df)
                if len(X) == 0:
                    return False
                
                input_size = X.shape[1]
                
                # 加载模型
                model = FastPredictor(input_size=input_size, hidden_size=128).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                
                # 缓存
                self.models[key] = model
                self.scalers[key] = scalers
                
                logger.info(f"Loaded model for {key}")
                return True
            except Exception as e:
                logger.error(f"Error loading model for {key}: {e}")
        
        # 训练新模型
        result = self.train_single_model(currency, period, lookback_days)
        return result is not None
    
    def predict_for_combination(self, currency: str, period: int, lookback_days: int) -> Optional[Dict]:
        """为特定组合和回测天数进行预测"""
        key = f"{currency}_{period}_{lookback_days}"
        
        # 确保模型已加载或训练
        if not self.load_or_train_model(currency, period, lookback_days):
            return None
        
        # 获取最新数据
        df = self.data_loader.get_recent_data(currency, period, lookback_days)
        
        if df.empty or len(df) < 50:
            return None
        
        try:
            model = self.models[key]
            scalers = self.scalers[key]
            
            # 准备特征
            X, _, feature_cols = self.prepare_features_simple(df)
            
            if len(X) == 0:
                return None
            
            # 使用最近的数据点
            X_recent = X[-1:].copy()
            
            # 标准化
            X_scaled = scalers['scaler_X'].transform(X_recent)
            
            # 预测
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
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
            # 简化计算：预期收益 = 预测利率 × 成交概率 × (1 - 风险调整)
            risk_adjustment = min(abs(trade_metrics['max_drawdown']) * 3, 0.5)
            expected_return = float(prediction * trade_metrics['trade_probability'] * (1 - risk_adjustment))
            
            # 计算综合得分
            stability_score = 1.0 - (trade_metrics['std_yield'] / (trade_metrics['avg_yield'] + 1e-6))
            stability_score = max(0, min(1, stability_score))
            
            score = float(expected_return * (1 + trade_metrics['sharpe_ratio'] / 20) * (1 + stability_score / 2))
            
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
                'score': score,
                'data_points': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {key}: {e}")
            return None
    
    def predict_all_lookback_periods(self, currency: str, period: int) -> Dict:
        """预测所有回测周期并计算综合得分"""
        all_predictions = []
        
        for lookback_days in self.config.lookback_days_list:
            result = self.predict_for_combination(currency, period, lookback_days)
            if result:
                all_predictions.append(result)
            else:
                # 如果机器学习模型失败，使用统计方法作为后备
                result = self.statistical_fallback(currency, period, lookback_days)
                if result:
                    all_predictions.append(result)
        
        if not all_predictions:
            return None
        
        # 计算加权综合得分
        # 权重：
        weights = {7: 4, 15: 6, 30: 4, 60: 2, 90: 1}
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
        
        # 获取当前收益率（使用最长周期的数据）
        current_yield = all_predictions[-1]['current_yield']
        
        # 获取其他指标的平均值
        avg_duration = float(np.mean([p['avg_duration_hours'] for p in all_predictions]))
        avg_sharpe = float(np.mean([p['sharpe_ratio'] for p in all_predictions]))
        avg_volatility = float(np.mean([p['volatility'] for p in all_predictions]))
        avg_max_drawdown = float(np.mean([p['max_drawdown'] for p in all_predictions]))
        
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
            'lookback_predictions': all_predictions,  # 保留所有周期的预测结果
            'total_data_points': int(sum([p['data_points'] for p in all_predictions])),
            'lookback_days_used': self.config.lookback_days_list
        }
    
    def find_optimal_combination(self, use_statistical_fallback: bool = True) -> Dict:
        """寻找最优组合，使用多个回测周期计算综合得分"""
        logger.info("Finding optimal combination with multiple lookback periods...")
        
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
            
            if not all_results:
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
                    'expected_return': optimal['composite_expected_return']
                },
                'detailed_metrics': optimal,
                'top_alternatives': all_results[1:5] if len(all_results) > 1 else [],
                'analysis_timestamp': datetime.now().isoformat(),
                'lookback_days_used': self.config.lookback_days_list,
                'total_combinations_evaluated': len(all_results),
                'status': 'success'
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error finding optimal combination: {e}")
            return {'status': 'error', 'message': str(e)}
        finally:
            self.data_loader.disconnect()
    
    def statistical_fallback(self, currency: str, period: int, lookback_days: int) -> Optional[Dict]:
        """统计方法后备方案"""
        df = self.data_loader.get_recent_data(currency, period, lookback_days)
        
        if df.empty or len(df) < 50:
            return None
        
        # 简单的统计方法：使用历史分位数
        yields = df['yield_rate'].values
        
        # 计算各种分位数
        q50 = float(np.percentile(yields, 50))  # 中位数
        q75 = float(np.percentile(yields, 75))  # 75分位数
        q90 = float(np.percentile(yields, 90))  # 90分位数
        
        # 计算每个分位数的成交概率
        prob_q50 = float(np.mean(yields <= q50))
        prob_q75 = float(np.mean(yields <= q75))
        prob_q90 = float(np.mean(yields <= q90))
        
        # 计算预期收益
        expected_q50 = float(q50 * prob_q50)
        expected_q75 = float(q75 * prob_q75)
        expected_q90 = float(q90 * prob_q90)
        
        # 选择预期收益最高的
        candidates = [
            (q50, prob_q50, expected_q50),
            (q75, prob_q75, expected_q75),
            (q90, prob_q90, expected_q90)
        ]
        
        best_yield, best_prob, best_expected = max(candidates, key=lambda x: x[2])
        
        # 计算其他指标
        trade_metrics = self.data_loader.calculate_trade_metrics(df, best_yield)
        
        return {
            'currency': currency,
            'period': period,
            'lookback_days': lookback_days,
            'current_yield': float(df['yield_rate'].iloc[-1]),
            'predicted_yield': float(best_yield),
            'trade_probability': float(best_prob),
            'expected_return': float(best_expected),
            'avg_duration_hours': trade_metrics['avg_duration_hours'],
            'sharpe_ratio': trade_metrics['sharpe_ratio'],
            'volatility': trade_metrics['volatility'],
            'max_drawdown': trade_metrics['max_drawdown'],
            'score': float(best_expected),
            'data_points': len(df),
            'method': 'statistical'
        }


# Flask API集成
from flask import Flask, jsonify, request

app = Flask(__name__)

# 全局配置
config = MLOptimizerConfig(
    db_path='data/lending_history.db',
    model_save_path='data/ml_models/',
    cache_dir='data/ml_cache/',
    periods=[2, 3, 4, 5, 6, 7, 10, 14, 15, 20, 30, 60, 90, 120],
    currencies=['fUST', 'fUSD'],
    lookback_days_list=[7, 15, 30, 60, 90],  # 多个回测周期
    min_records=20,
    batch_size=4096,
    n_epochs=1000,
    learning_rate=0.001,
    use_gpu=True
)

# 创建优化器实例
optimizer = SimpleGPUOptimizer(config)

@app.route('/api/ml/optimize', methods=['GET'])
def optimize():
    """机器学习优化接口"""
    try:
        force_retrain = request.args.get('force_retrain', 'false').lower() == 'true'
        use_fallback = request.args.get('use_fallback', 'true').lower() == 'true'
        download_history = request.args.get('download', 'false').lower() == 'true'
        
        if force_retrain:
            # 清理模型缓存
            optimizer.models.clear()
            optimizer.scalers.clear()
            
            # 删除模型文件
            import shutil
            if os.path.exists(config.model_save_path):
                shutil.rmtree(config.model_save_path)
                os.makedirs(config.model_save_path, exist_ok=True)
                
        if download_history:
            # 下载最新数据
            from funding_history_downloader import main as downloader
            downloader()
        
        result = optimizer.find_optimal_combination(use_statistical_fallback=use_fallback)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ml/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'gpu_available': torch.cuda.is_available(),
        'database_exists': os.path.exists(config.db_path),
        'config': {
            'currencies': config.currencies,
            'periods': config.periods,
            'lookback_days_used': config.lookback_days_list
        }
    })

@app.route('/api/ml/simple_optimize', methods=['GET'])
def simple_optimize():
    """简化优化接口 - 只返回最优组合"""
    try:
        result = optimizer.find_optimal_combination(use_statistical_fallback=True)
        
        if result['status'] == 'success':
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
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def main():
    """主函数"""
    print("=" * 70)
    print("🚀 Simple Machine Learning Optimizer for Bitfinex Lending")
    print("=" * 70)
    
    # 检查数据库
    if not os.path.exists(config.db_path):
        print(f"❌ Database not found: {config.db_path}")
        print("Please run the downloader first to collect data.")
        return
    
    print(f"✅ Database found: {config.db_path}")
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name}")
        print(f"   Memory: {gpu_memory:.2f} GB")
    else:
        print("⚠️  No GPU detected, using CPU")
    
    print("\n📊 Configuration:")
    print(f"   Currencies: {config.currencies}")
    print(f"   Periods: {config.periods}")
    print(f"   Lookback days: {config.lookback_days_list}")
    print(f"   Minimum records: {config.min_records}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Training epochs: {config.n_epochs}")

    print("\n🚀 Starting downloading data...")
    
    # 下载最新数据
    try:
        from funding_history_downloader import main as downloader
        downloader()
    except ImportError:
        logger.warning("Downloader module not found, skipping data download")
    
    print("\n🚀 Starting optimization...")
    
    # 运行优化
    result = optimizer.find_optimal_combination(use_statistical_fallback=True)
    
    if result['status'] == 'success':
        optimal = result['optimal_combination']
        detailed = result['detailed_metrics']
        
        print("\n" + "=" * 70)
        print("🎯 OPTIMAL COMBINATION FOUND")
        print("=" * 70)
        print(f"Currency: {optimal['currency']}")
        print(f"Period: {optimal['period']} days")
        print(f"Rate: {optimal['rate']:.4f}%")
        print(f"Trade Probability: {optimal['trade_probability']:.2%}")
        print(f"Expected Return: {optimal['expected_return']:.4f}")
        print()
        print("📈 Detailed Metrics:")
        print(f"   Current Yield: {detailed['current_yield']:.4f}%")
        print(f"   Composite Predicted Yield: {detailed['composite_predicted_yield']:.4f}%")
        print(f"   Composite Score: {detailed['composite_score']:.4f}")
        print(f"   Avg Sharpe Ratio: {detailed['avg_sharpe_ratio']:.3f}")
        print(f"   Avg Volatility: {detailed['avg_volatility']:.4f}")
        print(f"   Avg Max Drawdown: {detailed['avg_max_drawdown']:.2%}")
        print(f"   Avg Duration: {detailed['avg_duration_hours']:.1f} hours")
        print(f"   Total Data Points: {detailed['total_data_points']:,}")
        print(f"   Lookback Periods Used: {config.lookback_days_list}")
        print("=" * 70)
        
        # 保存结果 - 使用自定义JSON编码器
        output_file = 'data/optimal_combination.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, cls=NumpyJSONEncoder)
        
        print(f"\n📄 Result saved to: {output_file}")
        
        # 保存简化的最优组合
        simple_output = {
            'currency': optimal['currency'],
            'period': optimal['period'],
            'rate': optimal['rate'],
            'timestamp': result['analysis_timestamp'],
            'composite_score': detailed['composite_score']
        }
        
        with open('data/optimal_simple.json', 'w') as f:
            json.dump(simple_output, f, indent=2, cls=NumpyJSONEncoder)
        
        print(f"📄 Simplified result saved to: optimal_simple.json")
        
    else:
        print(f"\n❌ Optimization failed: {result.get('message', 'Unknown error')}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--api':
        # 启动Flask API
        print("Starting ML Optimizer API on http://localhost:5000")
        print("Available endpoints:")
        print("  GET /api/ml/optimize - 获取最优组合（完整版）")
        print("  GET /api/ml/simple_optimize - 获取最优组合（简化版）")
        print("  GET /api/ml/health - 健康检查")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    elif len(sys.argv) > 1 and sys.argv[1] == '--simple':
        # 只运行简单优化
        import requests
        response = requests.get('http://localhost:5000/api/ml/simple_optimize')
        print(json.dumps(response.json(), indent=2))
    else:
        # 直接运行优化
        main()
