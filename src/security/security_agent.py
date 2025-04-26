import os
import logging
import subprocess # 使用 subprocess 替代 os.system，更安全且功能更强大
import platform # 用于判断操作系统，以便使用正确的 ping 命令
from typing import Dict, Any, List, Optional, Union
import json
import random # 用于模拟一些随机的网络信息
import re # 用于简单解析 nmap 输出

# LangChain 相关的导入
from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_structured_chat_agent, AgentExecutor
# 根据你使用的 LLM 类型选择导入
# from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
# 如果使用其他模型，请导入相应的类

try:
    from database import SecurityIncidentDB, DATABASE_FILE
except ImportError:
    print("错误：无法导入 database.py。请确保 database.py 文件存在且在同一目录下。")
    exit()

try:
    from detector import predict_bash_script_danger, train_bash_detector, SAFE_SCRIPTS, DANGEROUS_SCRIPTS
    BASH_DETECTOR_MODEL_EXISTS = os.path.exists("ml_models/bash_detector_model.joblib")
    # 在 __main__ 块中处理模型训练
except ImportError:
    print("警告：无法导入 detector.py。Bash 脚本检测功能将不可用。")
    predict_bash_script_danger = None
    BASH_DETECTOR_MODEL_EXISTS = False

# --- 配置 ---
LOG_FILE = "security_agent.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- 定义 Agent 可以使用的工具 ---

# Pydantic 模型定义工具的输入 schema
from pydantic import BaseModel, Field

# 现有的工具输入模型
class AddIncidentInput(BaseModel):
    """Input schema for adding a security incident to the database."""
    incident_type: str = Field(description="Type of the security incident (e.g., 'Intrusion Attempt', 'Malware Detected', 'Port Scan', 'Suspicious Activity').")
    severity: str = Field(description="Severity of the incident ('Low', 'Medium', 'High', 'Critical'). Based on analysis and network context.")
    description: str = Field(description="Detailed description of the incident, summarizing the key findings and network context.")
    source_ip: Optional[str] = Field(None, description="Source IP address involved in the incident, if applicable.")
    source_port: Optional[int] = Field(None, description="Source port involved, if applicable.")
    dest_ip: Optional[str] = Field(None, description="Destination IP address involved, if applicable.")
    dest_port: Optional[int] = Field(None, description="Destination port involved, if applicable.")
    protocol: Optional[str] = Field(None, description="Protocol used (e.g., 'TCP', 'UDP', 'ICMP'), if applicable.")
    status: str = Field('Open', description="Status of the incident ('Open', 'Investigating', 'Closed', 'False Positive'). Default is 'Open'.")

class CheckScriptInput(BaseModel):
    """Input schema for checking a Bash script for dangerous behavior."""
    script_content: str = Field(description="The full content of the Bash script as a string.")

class RetrieveIncidentsInput(BaseModel):
    """Input schema for retrieving past security incidents from the database."""
    source_ip: Optional[str] = Field(None, description="Filter by source IP address.")
    dest_ip: Optional[str] = Field(None, description="Filter by destination IP address.")
    incident_type: Optional[str] = Field(None, description="Filter by incident type.")
    severity: Optional[str] = Field(None, description="Filter by severity ('Low', 'Medium', 'High', 'Critical').")
    status: Optional[str] = Field(None, description="Filter by status ('Open', 'Investigating', 'Closed', 'False Positive').")
    limit: int = Field(10, description="Maximum number of incidents to retrieve.")

# 新增的网络结构查询工具输入模型 (模拟保留 + 真实交互)
class GetDeviceInfoInput(BaseModel):
    """Input schema for getting device information based on IP address from simulated data."""
    ip_address: str = Field(description="The IP address of the device to query.")

class GetConnectionsInput(BaseModel):
    """Input schema for getting network connections for a given IP address from simulated data."""
    ip_address: str = Field(description="The IP address of the device to query for connections.")

class FindAttackPathsInput(BaseModel):
    """Input schema for finding potential attack paths between two IP addresses using simulated data."""
    source_ip: str = Field(description="The source IP address.")
    target_ip: str = Field(description="The target IP address.")

# 新增的真实网络交互工具输入模型
class PingHostInput(BaseModel):
    """Input schema for pinging a host to check reachability."""
    ip_address: str = Field(description="The IP address or hostname to ping.")
    count: int = Field(1, description="Number of ping packets to send (default 1).") # LLM can specify count

class NmapScanInput(BaseModel):
    """Input schema for performing an Nmap port scan."""
    ip_address: str = Field(description="The IP address or hostname to scan.")
    ports: Optional[str] = Field(None, description="Optional: Specific ports or range to scan (e.g., '22,80,443', '1-1024'). If not specified, Nmap's default scan (top 1000 common ports) will be performed.")
    arguments: Optional[str] = Field(None, description="Optional: Additional Nmap arguments (e.g., '-sV' for service version detection, '-O' for OS detection). Use with caution.") # Allow LLM some flexibility, but risky


# --- 工具函数 (包括模拟和真实的网络交互工具) ---

# 数据库工具 (与之前相同)
@tool
def add_security_incident(incident_data: AddIncidentInput) -> str:
    """
    Adds a new security incident to the database.
    Use this tool when you have analyzed the input and determined that a security incident needs to be recorded.
    Provide all necessary incident details as a structured input.
    """
    logging.info(f"Agent called add_security_incident tool with data: {incident_data.model_dump()}")
    db = None
    try:
        db = SecurityIncidentDB(DATABASE_FILE)
        incident_id = db.add_incident(**incident_data.model_dump(exclude_none=True))
        if incident_id is not None:
            logging.info(f"Successfully added incident with ID: {incident_id}")
            return f"Security incident successfully recorded with ID: {incident_id}"
        else:
            logging.error("Failed to add security incident to database.")
            return "Failed to record security incident."
    except Exception as e:
        logging.error(f"Error in add_security_incident tool: {e}")
        return f"An error occurred while recording the incident: {e}"
    finally:
        if db:
            db.close()

# Bash 检测工具 (与之前相同，包含可用性检查)
if predict_bash_script_danger and BASH_DETECTOR_MODEL_EXISTS:
    @tool
    def check_bash_script(input: CheckScriptInput) -> str:
        """
        Checks a given Bash script string for potentially dangerous behavior using a machine learning model.
        Input should be the full content of the script.
        Returns 'safe' or 'dangerous'.
        """
        logging.info("Agent called check_bash_script tool.")
        script_content = input.script_content
        if not script_content or not isinstance(script_content, str):
             logging.warning("check_bash_script received invalid input.")
             return "Error: Invalid script content provided."

        prediction = predict_bash_script_danger(script_content)

        if prediction is None:
            logging.error("Bash script detection failed.")
            return "Bash script detection failed."
        else:
            logging.info(f"Bash script prediction: {prediction}")
            return f"Bash script analysis result: {prediction}"
else:
    @tool
    def check_bash_script(input: CheckScriptInput) -> str:
        """
        Checks a given Bash script string for potentially dangerous behavior.
        This tool is currently unavailable because the required model is missing or the module could not be loaded.
        """
        return "Bash script analysis tool is currently unavailable."

# 数据库查询工具 (与之前相同，包含内存过滤的简化实现)
@tool
def retrieve_security_incidents(query: RetrieveIncidentsInput) -> str:
    """
    Retrieves past security incidents from the database based on specified criteria.
    Use this tool to get context about previous events related to the current input.
    You can filter by source_ip, dest_ip, incident_type, severity, or status.
    Returns a list of matching incidents as a JSON string, or an empty list if none found.
    Limit the number of results retrieved to avoid overwhelming the response.
    """
    logging.info(f"Agent called retrieve_security_incidents tool with query: {query.model_dump()}")
    db = None
    try:
        db = SecurityIncidentDB(DATABASE_FILE)
        all_incidents = db.get_all_incidents()

        filtered_incidents = []
        for inc in all_incidents:
            match = True
            # 检查每个过滤条件
            if query.source_ip is not None and inc['source_ip'] != query.source_ip:
                match = False
            if query.dest_ip is not None and inc['dest_ip'] != query.dest_ip:
                match = False
            if query.incident_type is not None and inc['incident_type'] != query.incident_type:
                match = False
            if query.severity is not None and inc['severity'] != query.severity:
                match = False
            if query.status is not None and inc['status'] != query.status:
                match = False

            if match:
                filtered_incidents.append({
                    'id': inc['id'],
                    'timestamp': inc['timestamp'],
                    'incident_type': inc['incident_type'],
                    'severity': inc['severity'],
                    'description': inc['description'][:100] + '...' if len(inc['description']) > 100 else inc['description'],
                    'source_ip': inc['source_ip'],
                    'dest_ip': inc['dest_ip'],
                    'status': inc['status']
                })

        filtered_incidents.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        limited_incidents = filtered_incidents[:query.limit]

        logging.info(f"Found {len(limited_incidents)} matching incidents.")
        return json.dumps(limited_incidents)

    except Exception as e:
        logging.error(f"Error in retrieve_security_incidents tool: {e}")
        return f"An error occurred while retrieving incidents: {e}"
    finally:
        if db:
            db.close()

# --- 模拟网络结构查询工具 (保留，用于补充真实工具无法提供的信息) ---

MOCK_NETWORK_DEVICES = {
    "192.168.1.1": {"type": "Router", "role": "Gateway", "os": "Cisco IOS"},
    "192.168.1.10": {"type": "Server", "role": "Web Server", "os": "Ubuntu Linux", "criticality": "High"},
    "192.168.1.20": {"type": "Workstation", "role": "User PC", "os": "Windows 10"},
    "192.168.1.100": {"type": "Server", "role": "Database Server", "os": "CentOS Linux", "criticality": "Critical"},
    "192.168.1.254": {"type": "Firewall", "role": "Internal Firewall", "os": "FortiOS"},
    "203.0.113.1": {"type": "Router", "role": "Internet Edge", "os": "Juniper Junos"},
    "203.0.113.5": {"type": "Unknown", "role": "External IP", "os": "Unknown"},
    "your_server_ip": {"type": "Server", "role": "SSH/Application Server", "os": "Ubuntu Linux", "criticality": "High"}
}

MOCK_NETWORK_CONNECTIONS = {
    "192.168.1.1": ["192.168.1.10", "192.168.1.20", "192.168.1.100", "192.168.1.254", "203.0.113.1"],
    "192.168.1.10": ["192.168.1.1", "192.168.1.100"],
    "192.168.1.20": ["192.168.1.1"],
    "192.168.1.100": ["192.168.1.1", "192.168.1.10"],
    "192.168.1.254": ["192.168.1.1", "203.0.113.1"],
    "203.0.113.1": ["192.168.1.1", "192.0.2.1"],
    "203.0.113.5": [],
    "your_server_ip": ["192.168.1.1", "external_internet"]
}

@tool
def get_simulated_device_info(input: GetDeviceInfoInput) -> str:
    """
    Retrieves simulated information about a network device based on its IP address from a static model.
    Provides details like device type, role, operating system, and criticality if available.
    Use this tool to get general context about an IP address when real-time scanning is not sufficient or appropriate.
    Returns device information as a JSON string, or 'Device not found in simulated model' if IP is unknown.
    """
    ip = input.ip_address
    logging.info(f"Agent called get_simulated_device_info tool for IP: {ip}")
    info = MOCK_NETWORK_DEVICES.get(ip)
    if info:
        logging.info(f"Found simulated device info: {info}")
        return json.dumps(info)
    else:
        logging.warning(f"Simulated device info not found for IP: {ip}")
        return "Device not found in simulated network model."

@tool
def get_simulated_connections(input: GetConnectionsInput) -> str:
    """
    Retrieves simulated network connections for a given IP address from a static model.
    Provides a list of other IP addresses or entities it is connected to in the simulated network model.
    Use this tool to understand the potential network neighborhood from static data.
    Returns a list of connected entities as a JSON string, or 'Connections not found in simulated model' if IP is unknown.
    """
    ip = input.ip_address
    logging.info(f"Agent called get_simulated_connections tool for IP: {ip}")
    connections = MOCK_NETWORK_CONNECTIONS.get(ip)
    if connections is not None:
        logging.info(f"Found simulated connections: {connections}")
        return json.dumps(connections)
    else:
        logging.warning(f"Simulated connections not found for IP: {ip}")
        return "Connections not found in simulated network model."

@tool
def find_simulated_attack_paths(input: FindAttackPathsInput) -> str:
    """
    Simulates finding potential attack paths between a source and target IP address using a simplified static model.
    This is a very basic simulation and does not perform real graph analysis.
    Use this tool for a high-level, static assessment of potential reachability.
    Returns a description of potential paths or indicates if no path is found in the simulation.
    """
    source_ip = input.source_ip
    target_ip = input.target_ip
    logging.info(f"Agent called find_simulated_attack_paths tool from {source_ip} to {target_ip}")

    source_exists = source_ip in MOCK_NETWORK_DEVICES
    target_info = MOCK_NETWORK_DEVICES.get(target_ip)
    target_exists = target_info is not None
    is_target_critical = target_info and target_info.get("criticality") in ["High", "Critical"]

    if source_exists and target_exists and is_target_critical:
        simulated_path = f"Simulated path found from {source_ip} to {target_ip}. Target is a critical server ({target_info.get('role')}). Potential attack vectors may exist through connected devices or known vulnerabilities based on the simulated model."
        logging.info(simulated_path)
        return simulated_path
    elif source_exists and target_exists:
         simulated_path = f"Simulated path analysis from {source_ip} to {target_ip}. Target is {target_info.get('role', 'an internal device')}. A direct path might exist based on the simulated model, but target is not marked as critical."
         logging.info(simulated_path)
         return simulated_path
    else:
        logging.warning(f"Cannot simulate path: Source ({source_ip}) or Target ({target_ip}) not in simulated network model.")
        return f"Simulated path analysis: Source ({source_ip}) or Target ({target_ip}) not found in the simulated network model, or no direct/critical path identified in simulation."


# --- 新增真实的网络交互工具 (使用 subprocess) ---

@tool
def ping_host(input: PingHostInput) -> str:
    """
    Pings a host (IP address or hostname) to check if it is reachable on the network.
    Use this tool to verify the live status of a device involved in a security event.
    Returns the output of the ping command, indicating success or failure.
    """
    ip_address = input.ip_address
    count = max(1, min(input.count, 5)) # Limit ping count to avoid abuse
    logging.info(f"Agent called ping_host tool for IP: {ip_address} with count {count}")

    # Determine ping command based on OS
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    command = ['ping', param, str(count), ip_address]

    try:
        # Use subprocess.run for better control and safety
        result = subprocess.run(command, capture_output=True, text=True, timeout=10) # 10 second timeout
        logging.info(f"Ping command: {' '.join(command)}")
        logging.info(f"Ping stdout:\n{result.stdout}")
        logging.info(f"Ping stderr:\n{result.stderr}")

        if result.returncode == 0:
            # Success, return relevant part of output
            return f"Ping successful for {ip_address}. Output:\n{result.stdout}"
        else:
            # Failure, return stderr and stdout
            return f"Ping failed for {ip_address}. Return code: {result.returncode}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"

    except FileNotFoundError:
        logging.error("Ping command not found. Is ping installed and in PATH?")
        return "Error: Ping command not found on the system."
    except subprocess.TimeoutExpired:
        logging.warning(f"Ping command timed out for {ip_address}.")
        return f"Error: Ping command timed out for {ip_address}."
    except Exception as e:
        logging.error(f"An error occurred during ping: {e}")
        return f"An error occurred during ping: {e}"

@tool
def nmap_scan(input: NmapScanInput) -> str:
    """
    Performs an Nmap scan on a target IP address or hostname.
    Can scan default ports or specified ports. Use this tool to discover open ports and services on a live host.
    **Requires Nmap to be installed and potentially root/administrator privileges.**
    Returns a summary of the Nmap scan results.
    """
    ip_address = input.ip_address
    ports = input.ports
    arguments = input.arguments

    logging.info(f"Agent called nmap_scan tool for IP: {ip_address}, Ports: {ports}, Args: {arguments}")

    # Basic command structure
    command = ['nmap', '-T4'] # -T4 is a common speed template (aggressive)

    if ports:
        command.extend(['-p', ports])

    if arguments:
        # WARNING: Allowing arbitrary arguments is risky!
        # A safer approach would be to have separate tools for -sV, -O etc.
        # For this example, we allow it but log a warning.
        logging.warning(f"Nmap scan allowing arbitrary arguments: {arguments}. This can be risky.")
        command.extend(arguments.split()) # Split arguments string into list

    command.append(ip_address)

    try:
        # Use subprocess.run
        # We capture stdout and stderr. Nmap sends progress to stderr.
        # Using -oN - sends normal output to stdout, which is easier to parse.
        command.extend(['-oN', '-'])
        result = subprocess.run(command, capture_output=True, text=True, timeout=60) # 60 second timeout for scan

        logging.info(f"Nmap command: {' '.join(command)}")
        logging.info(f"Nmap stdout:\n{result.stdout}")
        logging.info(f"Nmap stderr:\n{result.stderr}") # Nmap often puts progress/errors here

        # Simple parsing of Nmap output for open ports
        open_ports_summary = "Open Ports:\n"
        found_open = False
        # Look for lines like "PORT     STATE SERVICE"
        port_line_pattern = re.compile(r'^(\d+)/(\w+)\s+(\w+)\s+(.+)$') # e.g., 22/tcp   open  ssh
        for line in result.stdout.splitlines():
            if "Nmap scan report for" in line:
                open_ports_summary = line + "\n" + open_ports_summary # Add target info
            elif port_line_pattern.match(line):
                 match = port_line_pattern.match(line)
                 port, proto, state, service = match.groups()
                 if state == 'open':
                     open_ports_summary += f"- {port}/{proto} ({service})\n"
                     found_open = True
            elif "Nmap done" in line:
                 open_ports_summary += line # Add summary line

        if not found_open:
             open_ports_summary += "No open ports found or scan inconclusive.\n"


        if result.returncode == 0:
            # Nmap finished, even if no ports found
            return f"Nmap scan completed for {ip_address}.\n{open_ports_summary}\nRaw Output Snippet (stderr):\n{result.stderr[:500]}..." # Include snippet of stderr for context
        else:
            # Nmap returned non-zero, likely an error or host down
             return f"Nmap scan failed or encountered issues for {ip_address}. Return code: {result.returncode}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"


    except FileNotFoundError:
        logging.error("Nmap command not found. Is Nmap installed and in PATH? (Requires root/admin privileges?)")
        return "Error: Nmap command not found on the system. Please install Nmap and ensure it's in the system's PATH and executable with necessary permissions."
    except subprocess.TimeoutExpired:
        logging.warning(f"Nmap scan command timed out for {ip_address}.")
        return f"Error: Nmap scan command timed out for {ip_address}."
    except Exception as e:
        logging.error(f"An error occurred during Nmap scan: {e}")
        return f"An error occurred during Nmap scan: {e}"


# 将所有工具放在一个列表中
# 优先放置更常用或更基础的工具，但 LLM 会根据 Prompt 和输入自行选择
tools = [
    add_security_incident,
    retrieve_security_incidents,
    check_bash_script,
    ping_host, # 新增真实网络工具
    nmap_scan, # 新增真实网络工具
    get_simulated_device_info, # 保留模拟工具作为补充
    get_simulated_connections,
    find_simulated_attack_paths
]

# --- 定义 Agent Prompt ---
# 提示词指导 Agent 的行为和如何使用工具
# 强调使用新的网络工具获取实时信息
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a highly skilled network security analysis agent. Your task is to analyze incoming information about potential security events, determine if they represent a real security incident that needs to be recorded, and use the available tools to assist in your analysis and logging.

Available tools:
{tools}

Use the tools strategically to:
1. Analyze specific components like Bash scripts if mentioned in the input, using `check_bash_script`.
2. Retrieve relevant past incidents from the database to get historical context (e.g., based on IP addresses), using `retrieve_security_incidents`.
3. **IMPORTANT: Use the REAL-TIME network interaction tools (`ping_host`, `nmap_scan`) when the event involves specific IP addresses or hostnames and you need to verify their current status or discover open services.** For example, if an alert mentions traffic from an IP, use `ping_host` to see if it's currently active, and `nmap_scan` to see what ports are open.
4. **Use the SIMULATED network context tools (`get_simulated_device_info`, `get_simulated_connections`, `find_simulated_attack_paths`) to get general, static information about devices and potential paths based on a predefined model, especially when real-time scanning is not feasible or sufficient.**
5. **Crucially, use the 'add_security_incident' tool to record any event that you determine is a legitimate security incident.**

When deciding if something is a security incident and needs logging, consider:
- The nature of the event (e.g., failed logins, malware alerts, scans, suspicious commands).
- The severity level based on the description, any tool outputs (from script analysis, database history, and especially the REAL-TIME network tools), and the criticality of involved devices (from simulated info).
- Whether it involves known malicious indicators (from description or tool outputs).
- The LIVE status and open services of involved hosts (from `ping_host` and `nmap_scan`).
- **Do NOT log routine or clearly non-malicious events.**
- **Do NOT log the same event multiple times.** You are processing new information.

If you decide to log an incident using 'add_security_incident', carefully extract all the required parameters (incident_type, severity, description, source_ip, dest_ip, ports, protocol) from the input and any tool outputs. **Include relevant network context obtained from BOTH real-time tools and simulated tools in the description to provide a richer record.** Adjust the severity based on the analysis, including the criticality of involved devices and the findings from live scans.

If you use a tool, explain why you are using it in your reasoning.
After using tools and performing your analysis, if you decide to log an incident, call the 'add_security_incident' tool.
If you decide it's not an incident, state your conclusion clearly.
If you log an incident, provide a brief summary of what was logged, including the assigned ID.

{agent_scratchpad}
"""),
    ("human", "{input}"),
])

# --- 创建 Agent ---

class SecurityAgent:
    """
    A LangChain-based agent for network security event detection and logging,
    with simulated and real-time network interaction capabilities.
    """
    def __init__(self, llm: Any, db_path: str = DATABASE_FILE):
        """
        Initializes the SecurityAgent.

        Args:
            llm: The LangChain LLM instance to use (e.g., ChatOpenAI, Ollama).
            db_path: Path to the SQLite database file.
        """
        self.llm = llm
        self.db_path = db_path
        self.tools = tools # Use the globally defined tools list

        # 检查 Bash Detector 模型状态并在初始化时打印警告
        if predict_bash_script_danger and not BASH_DETECTOR_MODEL_EXISTS:
             print("警告: Bash Detector 模型文件未找到。Bash 脚本检测工具将返回错误。")
             logging.warning("Bash Detector model files not found during agent initialization.")
        elif not predict_bash_script_danger:
             print("警告: bash_detector.py 模块未加载成功。Bash 脚本检测工具将返回错误。")
             logging.warning("bash_detector.py module failed to load.")

        # 创建 Agent
        self.agent = create_structured_chat_agent(self.llm, self.tools, prompt)

        # 创建 Agent Executor
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        logging.info("SecurityAgent initialized.")

    def run_analysis(self, event_data: Dict[str, Any]) -> str:
        """
        Runs the security analysis agent on the given event data.

        Args:
            event_data: A dictionary containing information about the potential security event.
                        The dictionary keys/values should be descriptive for the LLM.
                        Include keys like 'source_ip', 'dest_ip', 'hostname', 'script_content'
                        to provide context for tool use.

        Returns:
            The final output from the Agent (a summary, confirmation of logging, or conclusion).
        """
        logging.info(f"Running analysis for event: {event_data}")
        input_string = "Analyze the following security event:\n"
        for key, value in event_data.items():
             input_string += f"{key}: {value}\n"

        try:
            result = self.agent_executor.invoke({"input": input_string})
            logging.info(f"Agent analysis complete. Result: {result}")
            return result['output']
        except Exception as e:
            logging.error(f"Error running agent analysis: {e}")
            return f"An error occurred during analysis: {e}"

# --- 示例用法 (在 __main__ 块中) ---

if __name__ == "__main__":
    # --- 准备 LLM ---
    # **重要：** 根据你使用的 LLM 取消注释并配置
    # 例如使用 OpenAI (需要设置 OPENAI_API_KEY 环境变量)
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # 或其他模型

    # 例如使用 Ollama (确保 Ollama 正在运行并已拉取模型，如 llama3)
    # 注意：较大的模型（如 llama3）通常在遵循指令和使用工具方面表现更好
    try:
        llm = Ollama(model="llama3") # 替换为你拉取的模型名称
        # 简单的测试，确保 LLM 可用
        llm.invoke("Hello, world!")
    except Exception as e:
        print(f"错误：无法初始化 LLM。请检查 Ollama 服务是否运行以及模型名称是否正确。错误信息: {e}")
        llm = None # 标记 LLM 不可用

    # 如果没有配置 LLM，退出
    if llm is None:
        print("错误：未成功配置或连接到 LLM。请检查你的 LLM 设置。")
        exit()

    # --- 检查并训练 Bash Detector (如果需要) ---
    if predict_bash_script_danger and not BASH_DETECTOR_MODEL_EXISTS:
         print("\n--- 训练 Bash 脚本检测模型 (示例数据) ---")
         print("警告: 使用示例数据训练的模型仅用于演示，准确性非常有限。")
         train_success = train_bash_detector(SAFE_SCRIPTS, DANGEROUS_SCRIPTS)
         if not train_success:
             print("错误：Bash Detector 模型训练失败，相关工具将无法正常工作。")
         else:
             BASH_DETECTOR_MODEL_EXISTS = True # 更新状态
         print("-" * 30)
    elif not predict_bash_script_danger:
         print("\n--- Bash Detector 模块未加载，Bash 脚本检测工具不可用 ---")
         print("-" * 30)

    # --- 检查 Nmap 和 Ping 命令是否存在 ---
    print("\n--- 检查网络工具可用性 ---")
    ping_available = False
    nmap_available = False
    try:
        subprocess.run(['ping', '-h'], capture_output=True, timeout=5)
        ping_available = True
        print("Ping command found.")
    except (FileNotFoundError, subprocess.SubprocessError):
        print("警告: Ping command not found or not executable. Ping tool will not work.")

    try:
        # Check Nmap version to see if it's executable
        subprocess.run(['nmap', '--version'], capture_output=True, timeout=5)
        nmap_available = True
        print("Nmap command found.")
        print("请注意：运行 Nmap 可能需要管理员/root 权限。")
    except (FileNotFoundError, subprocess.SubprocessError):
        print("警告: Nmap command not found or not executable. Nmap scan tool will not work.")
        print("请确保 Nmap 已安装，并且运行脚本的用户有权限执行它（可能需要管理员/root 权限）。")
    print("-" * 30)


    # --- 初始化 Agent ---
    security_agent = SecurityAgent(llm=llm)

    # --- 运行分析示例 (包含利用真实网络工具) ---

    print("\n--- 示例 1: 分析来自外部 IP 的端口扫描警报 (Agent 可能会 Ping 和 Nmap 扫描源 IP) ---")
    # 注意：扫描外部 IP 可能违反政策，这里仅作示例。在实际应用中请谨慎！
    # 建议将目标 IP 改为内部可控的测试 IP 或 localhost 进行测试。
    event_data_1 = {
        'type': 'IDS Alert',
        'message': 'Nmap scan detected from IP 203.0.113.5 targeting 192.168.1.100 on port 22.',
        'source_ip': '203.0.113.5', # 示例外部 IP
        'dest_ip': '192.168.1.100', # 示例内部 IP (请根据你的模拟/测试环境修改)
        'dest_port': 22,
        'protocol': 'TCP'
    }
    print(f"发送事件数据给 Agent:\n{json.dumps(event_data_1, indent=2)}")
    analysis_result_1 = security_agent.run_analysis(event_data_1)
    print(f"\nAgent 分析结果:\n{analysis_result_1}")
    print("=" * 50)

    print("\n--- 示例 2: 分析内部主机之间的可疑流量 (Agent 可能会 Ping 源/目标 IP 并 Nmap 扫描目标 IP) ---")
    event_data_2 = {
        'type': 'Network Flow Anomaly',
        'message': 'Unusual high volume traffic detected from 192.168.1.20 to 192.168.1.100 on port 5432 (PostgreSQL).',
        'source_ip': '192.168.1.20', # 示例内部 IP
        'dest_ip': '192.168.1.100', # 示例内部 IP (请根据你的模拟/测试环境修改)
        'dest_port': 5432,
        'protocol': 'TCP'
    }
    print(f"发送事件数据给 Agent:\n{json.dumps(event_data_2, indent=2)}")
    analysis_result_2 = security_agent.run_analysis(event_data_2)
    print(f"\nAgent 分析结果:\n{analysis_result_2}")
    print("=" * 50)

    print("\n--- 示例 3: 分析涉及关键服务器的失败登录尝试 (Agent 可能会 Ping 目标 IP 并获取模拟设备信息) ---")
    event_data_3 = {
        'type': 'Authentication Log',
        'message': 'Multiple failed login attempts for user "admin" on server 192.168.1.100 via SSH.',
        'source_ip': '192.168.1.20', # 假设来自内部 IP
        'dest_ip': '192.168.1.100', # 示例内部 IP (请根据你的模拟/测试环境修改)
        'dest_port': 22,
        'protocol': 'TCP',
        'user': 'admin'
    }
    print(f"发送事件数据给 Agent:\n{json.dumps(event_data_3, indent=2)}")
    analysis_result_3 = security_agent.run_analysis(event_data_3)
    print(f"\nAgent 分析结果:\n{analysis_result_3}")
    print("=" * 50)


    print("\n--- 示例 4: 分析可疑 Bash 脚本 (如果 Bash Detector 可用) ---")
    if predict_bash_script_danger and BASH_DETECTOR_MODEL_EXISTS:
        event_data_4 = {
            'type': 'User Activity',
            'message': 'User executed a suspicious command. Script content: #!/bin/bash\\nls -l /root && cat /etc/passwd', # 一个可能用于侦察的脚本
            'user': 'attacker',
            'source_ip': '192.168.1.20', # 示例内部 IP
            'script_content': '#!/bin/bash\nls -l /root && cat /etc/passwd'
        }
        print(f"发送事件数据给 Agent:\n{json.dumps(event_data_4, indent=2)}")
        analysis_result_4 = security_agent.run_analysis(event_data_4)
        print(f"\nAgent 分析结果:\n{analysis_result_4}")
    else:
        print("\n--- 跳过示例 4：Bash Detector 不可用或模型不存在 ---")
    print("=" * 50)


    print("\n--- 示例 5: 分析非安全事件 (应该不会被记录，Agent 可能不会使用网络工具) ---")
    event_data_5 = {
        'type': 'System Info',
        'message': 'CPU usage high on server 192.168.1.10.',
        'server_ip': '192.168.1.10', # 示例内部 IP
        'metric': 'CPU Usage'
    }
    print(f"发送事件数据给 Agent:\n{json.dumps(event_data_5, indent=2)}")
    analysis_result_5 = security_agent.run_analysis(event_data_5)
    print(f"\nAgent 分析结果:\n{analysis_result_5}")
    print("=" * 50)


    # 你现在可以检查 security_incidents.db 文件，应该会看到 Agent 记录的事件。
    # 可以使用 database.py 的 get_all_incidents 功能来验证：
    # db_checker = SecurityIncidentDB()
    # print("\n--- 数据库中的所有事件 ---")
    # all_incidents = db_checker.get_all_incidents()
    # for inc in all_incidents:
    #     print(inc)
    # db_checker.close()

