import pandas as pd
import sqlite3
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

# 导入 database.py 中的数据库接口类和数据库文件路径
try:
    from database import SecurityIncidentDB, DATABASE_FILE
except ImportError:
    print("错误：无法导入 database.py。请确保 database.py 文件存在且在同一目录下。")
    # 在实际应用中，这里可能需要更复杂的错误处理或退出
    exit()

# --- 配置 ---
LOG_FILE = "analysis.log"
# 配置日志记录
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 模型文件路径
MODEL_FILE = "severity_predictor_model.joblib"

# --- 数据加载函数 ---

def load_incidents_to_dataframe(db_path: str = DATABASE_FILE) -> pd.DataFrame:
    """
    从数据库加载所有事件到 Pandas DataFrame。

    Args:
        db_path: SQLite 数据库文件路径。

    Returns:
        包含事件数据的 Pandas DataFrame。如果加载失败或数据库为空，则返回空的 DataFrame。
    """
    logging.info(f"尝试从数据库加载数据到 DataFrame: {db_path}")
    try:
        # 直接使用 pandas 读取 SQLite 数据库
        conn = sqlite3.connect(db_path)
        # 检查表是否存在，避免 pd.read_sql_query 在表不存在时抛出异常
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='incidents';")
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            logging.warning(f"数据库文件 {db_path} 中不存在 'incidents' 表。返回空 DataFrame。")
            conn.close()
            return pd.DataFrame()

        df = pd.read_sql_query("SELECT * FROM incidents", conn)
        conn.close()

        if df.empty:
            logging.info("数据库中没有事件数据。返回空 DataFrame。")
            return df

        # 转换 timestamp 列为 datetime 对象，处理可能的转换错误
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            logging.error(f"转换 timestamp 列错误: {e}")
            # 如果转换失败，可以根据需要选择保留原样、删除列或抛出异常
            # 这里选择记录错误并继续，不转换该列
            pass


        logging.info(f"成功加载 {len(df)} 条事件数据。")
        return df
    except FileNotFoundError:
        logging.error(f"数据库文件未找到: {db_path}")
        print(f"错误：数据库文件未找到: {db_path}")
        return pd.DataFrame()
    except sqlite3.Error as e:
        logging.error(f"加载数据到 DataFrame 错误: {e}")
        print(f"错误：加载数据到 DataFrame 错误: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"加载数据到 DataFrame 时发生意外错误: {e}")
        print(f"错误：加载数据到 DataFrame 时发生意外错误: {e}")
        return pd.DataFrame()

# --- 分析函数 (返回结果) ---

def analyze_incident_types(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    统计不同事件类型的数量。

    Args:
        df: 包含事件数据的 DataFrame。

    Returns:
        一个 Pandas Series，索引是事件类型，值是对应的数量。如果 DataFrame 为空，返回 None。
    """
    if df.empty:
        logging.warning("analyze_incident_types: 输入 DataFrame 为空。")
        return None

    logging.info("开始分析事件类型。")
    if 'incident_type' not in df.columns:
         logging.error("analyze_incident_types: DataFrame 缺少 'incident_type' 列。")
         return None
    type_counts = df['incident_type'].value_counts()
    logging.info("事件类型分析完成。")
    return type_counts

def analyze_time_trends(df: pd.DataFrame) -> Optional[Dict[str, pd.Series]]:
    """
    分析事件发生的时间趋势（按天、按小时）。

    Args:
        df: 包含事件数据的 DataFrame。

    Returns:
        一个字典，包含 'daily' 和 'hourly' 两个 Pandas Series。如果 DataFrame 为空或 timestamp 列无效，返回 None。
    """
    if df.empty:
        logging.warning("analyze_time_trends: 输入 DataFrame 为空。")
        return None

    if 'timestamp' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        logging.error("analyze_time_trends: DataFrame 缺少有效的 'timestamp' 列 (datetime 格式)。")
        return None

    logging.info("开始分析时间趋势。")

    # 按天统计
    df_valid_time = df.dropna(subset=['timestamp']).copy() # 过滤掉无效时间戳
    if df_valid_time.empty:
        logging.warning("analyze_time_trends: 过滤掉无效时间戳后 DataFrame 为空。")
        return None

    df_valid_time['date'] = df_valid_time['timestamp'].dt.date
    daily_counts = df_valid_time['date'].value_counts().sort_index()

    # 按小时统计 (一天中的哪个小时事件最多)
    df_valid_time['hour'] = df_valid_time['timestamp'].dt.hour
    hourly_counts = df_valid_time['hour'].value_counts().sort_index()

    logging.info("时间趋势分析完成。")
    return {'daily': daily_counts, 'hourly': hourly_counts}

def analyze_correlation(df: pd.DataFrame) -> Optional[Dict[str, pd.DataFrame]]:
    """
    进行关联性分析 (例如，IP 地址与事件类型，或源/目标 IP 对)。

    Args:
        df: 包含事件数据的 DataFrame。

    Returns:
        一个字典，包含不同的关联性分析结果 DataFrame。
        例如：{'ip_type': DataFrame, 'ip_pair': DataFrame}。如果 DataFrame 为空或缺少必要列，返回 None。
    """
    if df.empty:
        logging.warning("analyze_correlation: 输入 DataFrame 为空。")
        return None

    results = {}
    logging.info("开始进行关联性分析。")

    # 1. IP 地址与事件类型关联
    if 'source_ip' in df.columns and 'incident_type' in df.columns:
        logging.info("分析 source_ip 与 incident_type 关联。")
        # 过滤掉 source_ip 为空的值
        ip_type_correlation = df[df['source_ip'].notna()].groupby('source_ip')['incident_type'].value_counts().unstack(fill_value=0)
        results['ip_type'] = ip_type_correlation
    else:
        logging.warning("analyze_correlation: 缺少 'source_ip' 或 'incident_type' 列，跳过 IP-类型关联分析。")

    # 2. 源 IP 和目标 IP 对的出现频率
    if 'source_ip' in df.columns and 'dest_ip' in df.columns:
        logging.info("分析 source_ip 与 dest_ip 对关联。")
        # 过滤掉 source_ip 或 dest_ip 为空的值
        ip_pair_correlation = df[df['source_ip'].notna() & df['dest_ip'].notna()].groupby(['source_ip', 'dest_ip']).size().reset_index(name='count')
        # 按计数排序，显示最常见的对
        ip_pair_correlation = ip_pair_correlation.sort_values(by='count', ascending=False)
        results['ip_pair'] = ip_pair_correlation
    else:
         logging.warning("analyze_correlation: 缺少 'source_ip' 或 'dest_ip' 列，跳过 IP 对关联分析。")


    logging.info("关联性分析完成。")
    return results if results else None # 如果没有任何分析结果，返回 None


def analyze_multi_factor(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    进行多因素分析 (例如，统计不同事件类型和严重性的组合)。

    Args:
        df: 包含事件数据的 DataFrame。

    Returns:
        一个 Pandas DataFrame，索引是事件类型，列是严重性，值是对应的组合数量。如果 DataFrame 为空或缺少必要列，返回 None。
    """
    if df.empty:
        logging.warning("analyze_multi_factor: 输入 DataFrame 为空。")
        return None

    if 'incident_type' not in df.columns or 'severity' not in df.columns:
        logging.error("analyze_multi_factor: DataFrame 缺少 'incident_type' 或 'severity' 列。")
        return None

    logging.info("开始进行多因素分析（类型与严重性）。")

    # 统计事件类型和严重性的组合数量
    type_severity_counts = df.groupby(['incident_type', 'severity']).size().unstack(fill_value=0)

    logging.info("多因素分析完成。")
    return type_severity_counts


def analyze_ports(df: pd.DataFrame) -> Optional[Dict[str, pd.Series]]:
    """
    进行端口分析 (例如，统计最常出现的源/目标端口，或高严重性事件相关的端口)。

    Args:
        df: 包含事件数据的 DataFrame。

    Returns:
        一个字典，包含不同端口分析结果的 Pandas Series。
        例如：{'source': Series, 'dest': Series, 'high_severity_dest': Series}。如果 DataFrame 为空或缺少必要列，返回 None。
    """
    if df.empty:
        logging.warning("analyze_ports: 输入 DataFrame 为空。")
        return None

    results = {}
    logging.info("开始进行端口分析。")

    # 源端口统计
    if 'source_port' in df.columns:
        logging.info("分析源端口。")
        source_port_counts = df[df['source_port'].notna()]['source_port'].value_counts()
        results['source'] = source_port_counts
    else:
        logging.warning("analyze_ports: DataFrame 缺少 'source_port' 列，跳过源端口分析。")

    # 目标端口统计
    if 'dest_port' in df.columns:
        logging.info("分析目标端口。")
        dest_port_counts = df[df['dest_port'].notna()]['dest_port'].value_counts()
        results['dest'] = dest_port_counts
    else:
        logging.warning("analyze_ports: DataFrame 缺少 'dest_port' 列，跳过目标端口分析。")

    # 示例：分析高严重性事件相关的目标端口
    if 'severity' in df.columns and 'dest_port' in df.columns:
         logging.info("分析高严重性事件相关的目标端口。")
         high_severity_ports = df[df['severity'].isin(['High', 'Critical']) & df['dest_port'].notna()]['dest_port'].value_counts()
         results['high_severity_dest'] = high_severity_ports
    else:
         logging.warning("analyze_ports: 缺少 'severity' 或 'dest_port' 列，跳过高严重性端口分析。")


    logging.info("端口分析完成。")
    return results if results else None # 如果没有任何分析结果，返回 None


def retrieve_specific_incident(incident_id: int, db_path: str = DATABASE_FILE) -> Optional[Dict[str, Any]]:
    """
    调回特定事件信息。

    Args:
        incident_id: 要查询的事件 ID。
        db_path: SQLite 数据库文件路径。

    Returns:
        事件记录字典，如果未找到或发生错误则返回 None。
    """
    logging.info(f"尝试调回事件 ID: {incident_id}")
    db = None
    try:
        db = SecurityIncidentDB(db_path)
        incident = db.get_incident_by_id(incident_id)
        if incident:
            logging.info(f"成功调回事件 ID: {incident_id}")
            return incident
        else:
            logging.warning(f"未找到 ID 为 {incident_id} 的事件。")
            return None
    except Exception as e:
        logging.error(f"调回事件 (ID: {incident_id}) 错误: {e}")
        return None
    finally:
        if db:
            db.close()


# --- 机器学习部分 (预测严重性) ---

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 定义用于训练和预测的特征
ML_FEATURES = ['incident_type', 'source_port', 'dest_port', 'protocol']
ML_TARGET = 'severity'
# 定义分类特征和数值特征
ML_CATEGORICAL_FEATURES = ['incident_type', 'protocol']
ML_NUMERICAL_FEATURES = ['source_port', 'dest_port']
# 定义严重性顺序，用于 LabelEncoder 的 classes_ 属性排序（可选，但有助于一致性）
# 注意：LabelEncoder 默认按字母顺序排序，如果需要特定顺序，需要手动映射或使用OrdinalEncoder
# 假设我们希望 Low < Medium < High < Critical
SEVERITY_ORDER = ['Low', 'Medium', 'High', 'Critical']


def train_severity_predictor(df: pd.DataFrame, model_file: str = MODEL_FILE) -> bool:
    """
    训练一个机器学习模型来预测事件严重性，并保存模型。

    Args:
        df: 包含事件数据的 DataFrame。
        model_file: 保存训练好的模型的文件路径。

    Returns:
        布尔值，表示训练是否成功。
    """
    if df.empty:
        print("没有数据可供训练模型。")
        logging.warning("train_severity_predictor: 输入 DataFrame 为空。")
        return False

    # 过滤掉目标变量为空的行
    df_train = df.dropna(subset=[ML_TARGET]).copy()

    if df_train.empty:
        print(f"没有有效的 '{ML_TARGET}' 数据可供训练。")
        logging.warning(f"train_severity_predictor: 没有有效的 '{ML_TARGET}' 数据可供训练。")
        return False

    # 检查是否有足够的类别进行分层抽样 (至少每个类别在训练/测试集中有1个样本)
    if df_train[ML_TARGET].nunique() < 2:
         print(f"'{ML_TARGET}' 列至少需要两个不同的类别才能训练分类模型。")
         logging.warning(f"train_severity_predictor: '{ML_TARGET}' 列类别少于2个。")
         return False

    target_counts = df_train[ML_TARGET].value_counts()
    if any(target_counts < 2): # 对于 test_size=0.2，需要至少2个样本才能保证训练/测试集都有该类别
         print(f"某些 '{ML_TARGET}' 类别的样本数量过少，无法进行分层抽样。请检查数据。")
         logging.warning(f"train_severity_predictor: 某些 '{ML_TARGET}' 类别的样本数量过少。")
         # 可以选择不使用 stratify，或者合并小类别，或者要求更多数据
         # 这里选择返回 False
         return False


    X = df_train[ML_FEATURES]
    y = df_train[ML_TARGET]

    # 处理缺失值：对于端口使用 -1 或其他标记，对于文本使用 'Unknown'
    # 确保在训练和预测时使用相同的缺失值填充策略
    X.loc[:, 'source_port'] = X['source_port'].fillna(-1)
    X.loc[:, 'dest_port'] = X['dest_port'].fillna(-1)
    X.loc[:, 'protocol'] = X['protocol'].fillna('Unknown')
    X.loc[:, 'incident_type'] = X['incident_type'].fillna('Unknown') # incident_type 应该是非空的，但以防万一


    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ML_NUMERICAL_FEATURES), # 数值特征直接通过
            ('cat', OneHotEncoder(handle_unknown='ignore'), ML_CATEGORICAL_FEATURES) # 分类特征独热编码
        ],
        remainder='passthrough' # 保留未指定的列 (这里应该没有，但以防万一)
    )

    # 编码目标变量 (严重性)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 尝试按照预定义的顺序重新排序 classes_，如果所有预定义类别都在数据中
    if all(item in label_encoder.classes_ for item in SEVERITY_ORDER):
        # 创建一个映射，将原始索引映射到目标顺序的索引
        order_map = {label: i for i, label in enumerate(SEVERITY_ORDER)}
        # 获取当前 classes_ 的排序索引
        current_order_indices = [list(label_encoder.classes_).index(label) for label in SEVERITY_ORDER]
        # 重新排序 classes_ 和 inverse_transform 查找表
        label_encoder.classes_ = label_encoder.classes_[current_order_indices]
        # 注意：y_encoded 已经是根据原始 fit_transform 生成的，不需要重新编码
        # 但 inverse_transform 会使用新的 classes_ 顺序

    print(f"严重性类别映射 (Encoded -> String): {dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))}")
    logging.info(f"严重性类别映射 (Encoded -> String): {dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))}")


    # 分割数据集
    # 使用 stratify 确保训练集和测试集中的类别分布相似
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # 创建模型管道：预处理 -> 模型
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

    # 训练模型
    logging.info("开始训练 RandomForestClassifier 模型。")
    try:
        model.fit(X_train, y_train)
        logging.info("模型训练完成。")
    except Exception as e:
        logging.error(f"模型训练失败: {e}")
        print(f"错误：模型训练失败: {e}")
        return False


    # 评估模型
    logging.info("开始评估模型。")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 确保 target_names 与 y_test 中的类别对应
    # LabelEncoder 的 classes_ 属性提供了编码值到原始类别的映射
    target_names = label_encoder.classes_

    try:
        report = classification_report(y_test, y_pred, target_names=target_names)
        print("\n--- 模型评估报告 ---")
        print(f"准确率: {accuracy:.2f}")
        print(report)
        logging.info(f"模型评估完成。准确率: {accuracy:.2f}")
        logging.info(f"分类报告:\n{report}")
    except ValueError as e:
        logging.warning(f"生成分类报告失败，可能测试集中缺少某些类别: {e}")
        print(f"警告：生成分类报告失败，可能测试集中缺少某些类别: {e}")
        print(f"准确率: {accuracy:.2f}")


    # 保存模型、LabelEncoder 和特征列表
    try:
        model_data = {
            'model': model,
            'label_encoder': label_encoder,
            'features': ML_FEATURES, # 保存用于训练的特征列表
            'categorical_features': ML_CATEGORICAL_FEATURES,
            'numerical_features': ML_NUMERICAL_FEATURES
        }
        joblib.dump(model_data, model_file)
        print(f"\n模型已保存到 {model_file}")
        logging.info(f"模型已保存到 {model_file}")
        return True
    except Exception as e:
        logging.error(f"保存模型错误: {e}")
        print(f"错误：保存模型错误: {e}")
        return False


def load_severity_predictor(model_file: str = MODEL_FILE) -> Optional[Dict[str, Any]]:
    """
    加载训练好的严重性预测模型。

    Args:
        model_file: 训练好的模型文件路径。

    Returns:
        包含模型、LabelEncoder 和特征列表的字典，如果加载失败则返回 None。
    """
    if not os.path.exists(model_file):
        logging.error(f"模型文件未找到: {model_file}")
        print(f"错误：模型文件未找到: {model_file}。请先运行 --train_ml 训练模型。")
        return None

    try:
        model_data = joblib.load(model_file)
        logging.info(f"成功加载模型文件: {model_file}")
        # 简单验证加载的数据是否包含预期键
        if all(key in model_data for key in ['model', 'label_encoder', 'features']):
             return model_data
        else:
             logging.error(f"加载的模型文件 {model_file} 格式不正确。")
             print(f"错误：加载的模型文件 {model_file} 格式不正确。")
             return None
    except Exception as e:
        logging.error(f"加载模型错误: {e}")
        print(f"错误：加载模型错误: {e}")
        return None


def predict_severity(incident_data: Union[Dict[str, Any], List[Dict[str, Any]]], model_file: str = MODEL_FILE) -> Optional[Union[str, List[str]]]:
    """
    使用训练好的模型预测一个或多个事件的严重性。

    Args:
        incident_data: 包含事件特征的字典 或 字典列表。
                       例如: {'incident_type': 'Port Scan', 'source_port': 12345, ...}
                       或 [{'incident_type': 'Port Scan', ...}, {'incident_type': 'Malware Detected', ...}]
        model_file: 训练好的模型文件路径。

    Returns:
        预测的严重性字符串 (如果输入是单个字典) 或字符串列表 (如果输入是字典列表)。
        如果加载模型失败或预测错误则返回 None。
    """
    model_data = load_severity_predictor(model_file)
    if model_data is None:
        return None # 加载模型失败

    model = model_data['model']
    label_encoder = model_data['label_encoder']
    features = model_data['features']
    # categorical_features = model_data['categorical_features'] # 备用，如果需要特殊处理

    # 准备预测数据
    is_single_incident = isinstance(incident_data, dict)
    if is_single_incident:
        input_list = [incident_data]
    elif isinstance(incident_data, list):
        input_list = incident_data
    else:
        logging.error("predict_severity: 输入数据格式不正确，应为字典或字典列表。")
        print("错误：预测输入数据格式不正确。")
        return None

    if not input_list:
        logging.warning("predict_severity: 输入数据列表为空。")
        return None

    # 将输入的字典/列表转换为 DataFrame
    input_df = pd.DataFrame(input_list)

    # 确保 DataFrame 包含模型训练时使用的所有特征列，并按相同顺序排列
    # 如果输入数据缺少某个特征，添加该列并用 None 填充
    for col in features:
        if col not in input_df.columns:
            input_df[col] = None

    input_df = input_df[features] # 确保列顺序

    # 填充缺失值，与训练时保持一致
    # 这里使用 .loc[] 避免 SettingWithCopyWarning
    input_df.loc[:, 'source_port'] = input_df['source_port'].fillna(-1)
    input_df.loc[:, 'dest_port'] = input_df['dest_port'].fillna(-1)
    input_df.loc[:, 'protocol'] = input_df['protocol'].fillna('Unknown')
    input_df.loc[:, 'incident_type'] = input_df['incident_type'].fillna('Unknown')


    # 进行预测
    try:
        predicted_encoded = model.predict(input_df)

        # 解码预测结果
        predicted_severity = label_encoder.inverse_transform(predicted_encoded)

        logging.info(f"成功预测 {len(predicted_severity)} 个事件的严重性。")

        if is_single_incident:
            return predicted_severity[0] # 返回单个字符串
        else:
            return predicted_severity.tolist() # 返回字符串列表

    except Exception as e:
        logging.error(f"预测严重性错误: {e}")
        print(f"错误：预测严重性错误: {e}")
        return None

# --- 辅助函数用于命令行接口打印结果 ---

def print_analysis_results(analysis_type: str, results: Any):
    """
    根据分析类型格式化并打印结果。
    """
    if results is None:
        print(f"\n--- {analysis_type.replace('_', ' ').title()} 分析结果 ---")
        print("没有结果可显示，可能数据不足或发生错误。")
        return

    print(f"\n--- {analysis_type.replace('_', ' ').title()} 分析结果 ---")
    if isinstance(results, pd.Series):
        print(results.to_string()) # 使用 to_string() 避免截断
    elif isinstance(results, pd.DataFrame):
        print(results.to_string())
    elif isinstance(results, dict):
        for key, value in results.items():
            print(f"\n--- {analysis_type.replace('_', ' ').title()} - {key.replace('_', ' ').title()} ---")
            if isinstance(value, (pd.Series, pd.DataFrame)):
                 print(value.to_string())
            else:
                 print(value) # 打印字典中的其他类型值
    else:
        print(results) # 打印其他类型的结果

# --- 主执行逻辑 (命令行接口) ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="网络安全事件数据库分析工具")
    parser.add_argument('--analyze', nargs='+', choices=['type', 'time', 'correlation', 'multifactor', 'ports', 'all'],
                        help="选择要执行的分析类型 ('type', 'time', 'correlation', 'multifactor', 'ports', 'all')")
    parser.add_argument('--get_id', type=int, help="根据事件 ID 查询特定事件信息")
    parser.add_argument('--train_ml', action='store_true', help="训练严重性预测机器学习模型")
    parser.add_argument('--predict_ml', nargs='+', help="使用训练好的模型预测一个或多个事件严重性。输入格式: key1=value1 key2=value2 ... 对于多个事件，使用 --predict_ml event1_key1=value1 event1_key2=value2 ... --predict_ml event2_key1=value1 ...")


    args = parser.parse_args()

    # 加载数据到 DataFrame (用于分析功能和训练 ML)
    # 只有当需要分析或训练ML时才加载数据
    df = pd.DataFrame()
    if args.analyze or args.train_ml:
         df = load_incidents_to_dataframe()
         if df.empty:
             print("警告：数据库中没有数据或加载失败，跳过分析和训练。")


    if args.analyze:
        if not df.empty:
            analysis_map = {
                'type': analyze_incident_types,
                'time': analyze_time_trends,
                'correlation': analyze_correlation,
                'multifactor': analyze_multi_factor,
                'ports': analyze_ports
            }
            if 'all' in args.analyze:
                for analysis_type, func in analysis_map.items():
                    results = func(df)
                    print_analysis_results(analysis_type, results)
            else:
                for analysis_type in args.analyze:
                    if analysis_type in analysis_map:
                        results = analysis_map[analysis_type](df)
                        print_analysis_results(analysis_type, results)
                    else:
                        print(f"未知分析类型: {analysis_type}")
        else:
            print("无法执行分析，因为没有加载到数据。")


    if args.get_id is not None:
        # retrieve_specific_incident 直接处理数据库连接
        incident = retrieve_specific_incident(args.get_id)
        print(f"\n--- 查询事件 ID: {args.get_id} ---")
        if incident:
             import pprint
             pprint.pprint(incident)
        else:
             print(f"未找到 ID 为 {args.get_id} 的事件。")


    if args.train_ml:
        if not df.empty:
            print("\n--- 训练机器学习模型 ---")
            train_severity_predictor(df)
        else:
             print("无法训练模型，因为没有加载到数据。")


    if args.predict_ml:
        if not args.predict_ml:
             print("请提供要预测的事件特征，例如: --predict_ml incident_type=Port\\ Scan source_port=12345")
        else:
            # 解析输入的键值对列表
            # 支持输入多个事件，每个事件是一组 key=value 对
            # 例如: --predict_ml type=scan ip=1.1.1.1 --predict_ml type=malware ip=2.2.2.2
            incident_list_to_predict = []
            current_incident_data = {}
            for item in args.predict_ml:
                if '=' in item:
                    key, value = item.split('=', 1)
                    # 尝试将值转换为数字，如果失败则保留字符串
                    try:
                        current_incident_data[key] = int(value)
                    except ValueError:
                         try:
                             current_incident_data[key] = float(value)
                         except ValueError:
                            current_incident_data[key] = value.replace('\\ ', ' ') # 处理命令行中的空格转义
                else:
                    # 如果遇到没有 '=' 的项，并且 current_incident_data 不为空，
                    # 认为这是一个新事件的开始，将前一个事件添加到列表
                    if current_incident_data:
                        incident_list_to_predict.append(current_incident_data)
                        current_incident_data = {}
                    # 处理没有 '=' 的项，可能是一个错误输入，或者作为下一个事件的第一个键？
                    # 为了简单，这里假设所有输入都是 key=value 对，或者最后一个事件结束没有明确分隔符
                    # 如果需要更严格的解析，可以修改这里的逻辑
                    print(f"警告: 跳过无法解析的输入项 '{item}'。请确保输入为 key=value 格式。")

            # 添加最后一个事件数据
            if current_incident_data:
                 incident_list_to_predict.append(current_incident_data)

            if not incident_list_to_predict:
                 print("未能解析输入的事件特征。请检查格式。")
            else:
                print("\n--- 严重性预测 ---")
                # 如果只有一个事件，传递字典；如果多个，传递列表
                if len(incident_list_to_predict) == 1:
                    incident_data_input = incident_list_to_predict[0]
                    print(f"输入特征: {incident_data_input}")
                else:
                    incident_data_input = incident_list_to_predict
                    print(f"输入特征列表 ({len(incident_data_input)}个事件):")
                    for i, data in enumerate(incident_data_input):
                        print(f"  事件 {i+1}: {data}")


                predicted = predict_severity(incident_data_input)

                if predicted is not None:
                    if isinstance(predicted, list):
                         print("\n预测的严重性为:")
                         for i, sev in enumerate(predicted):
                             print(f"  事件 {i+1}: {sev}")
                    else: # 单个预测结果
                         print(f"\n预测的严重性为: {predicted}")
                else:
                    print("预测失败。")

    # 注意：SecurityIncidentDB 实例只在 get_id 中创建和关闭，
    # load_incidents_to_dataframe 使用 pandas 内部管理连接。
    # 所以这里不需要额外的 db.close() 调用，除非你在 __main__ 中创建了全局 db 实例。
    # 当前设计下，每个需要 db 连接的函数自己管理连接是更安全的做法。
