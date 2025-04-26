import os
import logging
import subprocess
import platform
from typing import Dict, Any, List, Optional, Union
import json
import re
import shlex # 用于更安全地分割命令和参数
import sys # 用于退出脚本

# 导入 requests 库用于互联网访问
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print("警告: requests 库未安装。互联网访问工具将不可用。请运行 'pip install requests'。")
    REQUESTS_AVAILABLE = False

# Rich 相关的导入
from rich.console import Console
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.live import Live # 用于实时显示 Agent 的思考过程 (可选)
from rich.status import Status # 用于显示正在进行的任务状态

# LangChain 相关的导入
from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_structured_chat_agent, AgentExecutor
# 根据你使用的 LLM 类型选择导入并配置
# from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama # 示例使用 Ollama

# 加载环境变量 (可选，如果你使用需要 API 密钥的 LLM)
# from dotenv import load_dotenv
# load_dotenv()

# --- 配置 ---
LOG_FILE = "ops_agent.log"
# 配置日志记录
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Rich Console 实例
console = Console()

# --- 定义 Agent 可以使用的工具 ---

# Pydantic 模型定义工具的输入 schema
from pydantic import BaseModel, Field

# Bash 脚本生成工具输入模型
class GenerateBashScriptInput(BaseModel):
    """Input schema for generating a Bash script."""
    task_description: str = Field(description="A clear description of the operational task the user wants to accomplish using a Bash script.")

# 新增的代码生成工具输入模型
class GenerateCodeInput(BaseModel):
    """Input schema for generating code in a specific programming language."""
    task_description: str = Field(description="A clear description of the programming task.")
    language: str = Field(description="The programming language for the code (e.g., 'python', 'javascript', 'java', 'c++', 'json', 'yaml').") # Added common config languages
    requirements: Optional[str] = Field(None, description="Any specific requirements or libraries to use.")

# 新增的文件操作工具输入模型
class ReadFileInput(BaseModel):
    """Input schema for reading the content of a file."""
    file_path: str = Field(description="The absolute or relative path to the file to read.")

class WriteFileInput(BaseModel):
    """Input schema for writing content to a file."""
    file_path: str = Field(description="The absolute or relative path to the file to write to.")
    content: str = Field(description="The content to write to the file.")
    append: bool = Field(False, description="If true, append to the file instead of overwriting it. If false, overwrite the file.")

class DeleteFileInput(BaseModel):
    """Input schema for deleting a file."""
    file_path: str = Field(description="The absolute or relative path to the file to delete.")

# 新增的互联网访问工具输入模型
class FetchUrlContentInput(BaseModel):
    """Input schema for fetching content from a URL."""
    url: str = Field(description="The URL to fetch content from (e.g., 'http://example.com', 'https://api.service.com/data').")
    timeout: int = Field(10, description="Timeout in seconds for the request.")

# 新增的系统信息工具输入模型 (通常不需要输入，但为了StructuredTool规范，可以定义一个空的或带可选参数的模型)
class GetSystemInfoInput(BaseModel):
    """Input schema for getting system information (optional, can be empty)."""
    detail_level: Optional[str] = Field("basic", description="Level of detail requested ('basic', 'full').")


# --- 工具函数 ---

# Bash 脚本生成工具 (内部调用 LLM)
@tool
def generate_bash_script(input: GenerateBashScriptInput) -> str:
    """
    Generates a Bash script based on the provided task description.
    Use this tool when the user asks you to perform an operational task that requires a script.
    Provide the full task description as input.
    The output will be the generated Bash script enclosed in ```bash ... ``` markers.
    """
    logging.info(f"Agent called generate_bash_script tool with task: {input.task_description}")

    # This tool uses the LLM again internally to generate the script
    # We create a specific prompt for this task
    script_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Bash scripting assistant. Your sole purpose is to generate a Bash script that precisely accomplishes the task described by the user.
Do NOT include any explanations or extra text outside the script block.
Wrap the generated script in ```bash and ``` markers.
Ensure the script is executable and follows best practices.
If the task is impossible or unsafe to script, state that clearly instead of generating a script.
Task: {task_description}"""),
        ("human", "{task_description}"),
    ])

    try:
        # Use the same LLM instance that the Agent uses
        script_chain = script_generation_prompt | llm | StrOutputParser() # Assuming llm is available in this scope
        generated_script = script_chain.invoke({"task_description": input.task_description})
        logging.info(f"Generated script:\n{generated_script}")
        return generated_script
    except Exception as e:
        logging.error(f"Error generating bash script: {e}")
        return f"Error generating bash script: {e}"

# 代码生成工具 (内部调用 LLM)
@tool
def generate_code(input: GenerateCodeInput) -> str:
    """
    Generates code in a specific programming language based on the task description and requirements.
    Use this tool when the user asks you to write code or configuration files.
    Provide the task description, language, and any requirements.
    The output will be the generated code enclosed in markdown code block markers (e.g., ```python ... ```).
    """
    logging.info(f"Agent called generate_code tool for task: {input.task_description}, language: {input.language}")

    code_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert programming assistant. Your sole purpose is to generate code in the specified language that precisely accomplishes the task described by the user.
Do NOT include any explanations or extra text outside the code block.
Wrap the generated code in markdown code block markers appropriate for the language (e.g., ```python, ```javascript, ```java, ```json, ```yaml).
Ensure the code is correct and follows best practices for the language.
If the task is impossible or unsafe, state that clearly instead of generating code.
Task: {task_description}
Language: {language}
Requirements: {requirements}"""),
        ("human", f"Task: {input.task_description}\nLanguage: {input.language}\nRequirements: {input.requirements}"),
    ])

    try:
        code_chain = code_generation_prompt | llm | StrOutputParser() # Assuming llm is available
        generated_code = code_chain.invoke({"task_description": input.task_description, "language": input.language, "requirements": input.requirements})
        logging.info(f"Generated code ({input.language}):\n{generated_code}")
        return generated_code
    except Exception as e:
        logging.error(f"Error generating code: {e}")
        return f"Error generating code: {e}"

# 文件读取工具
@tool
def read_file(input: ReadFileInput) -> str:
    """
    Reads the content of a file from the local filesystem.
    Use this tool when you need to inspect the content of a file to understand its configuration or data.
    Provide the file path.
    Returns the file content as a string, or an error message if the file cannot be read.
    """
    file_path = os.path.expanduser(input.file_path) # Expand user home directory (~)
    logging.info(f"Agent called read_file tool for path: {file_path}")

    # Basic path validation to prevent reading system critical files (can be expanded)
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return f"Error: File not found at {file_path}"
    if not os.path.isfile(file_path):
         logging.warning(f"Path is not a file: {file_path}")
         return f"Error: Path {file_path} is not a file."
    # Add more robust checks if needed, e.g., prevent reading /etc/*, /dev/* etc.
    # This is a simple example, a real agent needs stricter path sanitization.

    try:
        with open(file_path, 'r') as f:
            content = f.read()
        logging.info(f"Successfully read file: {file_path}")
        # Truncate content for logging if very large
        log_content = content[:500] + '...' if len(content) > 500 else content
        logging.debug(f"File content read:\n{log_content}")
        return f"File content of '{file_path}':\n```\n{content}\n```" # Wrap in markdown code block for clarity
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return f"Error reading file {file_path}: {e}"

# 文件写入工具 (需要用户确认)
@tool
def write_file(input: WriteFileInput) -> str:
    """
    Writes content to a file on the local filesystem.
    **This action requires user confirmation.**
    Use this tool when you need to create or modify a file, e.g., writing configuration, scripts, or code.
    Provide the file path, content, and whether to append or overwrite.
    Returns a status message indicating success, failure, or user cancellation.
    """
    file_path = os.path.expanduser(input.file_path) # Expand user home directory (~)
    logging.info(f"Agent called write_file tool for path: {file_path}, append: {input.append}")

    mode = 'a' if input.append else 'w'
    action = "append to" if input.append else "write to"

    console.print(Panel(f"Agent proposes to {action} file: [bold blue]{file_path}[/bold blue]", title="[bold yellow]Action Required: Confirm File Write[/bold yellow]", expand=False))
    console.print(Syntax(input.content, "text", theme="default", line_numbers=True)) # Display content with basic highlighting

    if Confirm.ask(f"Do you want to confirm writing to {file_path} ({action})?"):
        try:
            # Ensure directory exists
            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.exists(dir_name):
                console.print(f"Directory '{dir_name}' does not exist. Creating...")
                os.makedirs(dir_name, exist_ok=True) # exist_ok prevents error if dir exists
                logging.info(f"Created directory: {dir_name}")

            with open(file_path, mode) as f:
                f.write(input.content)
            logging.info(f"Successfully wrote to file: {file_path} ({action})")
            console.print(f"[bold green]Successfully wrote to {file_path}[/bold green]")
            return f"Successfully wrote to file: {file_path} ({action})."
        except Exception as e:
            logging.error(f"Error writing to file {file_path}: {e}")
            console.print(f"[bold red]Error writing to file {file_path}: {e}[/bold red]")
            return f"Error writing to file {file_path}: {e}"
    else:
        logging.warning(f"User cancelled writing to file: {file_path}")
        console.print(f"[bold yellow]File write to {file_path} cancelled by user.[/bold yellow]")
        return f"File write to {file_path} cancelled by user."

# 文件删除工具 (需要用户确认)
@tool
def delete_file(input: DeleteFileInput) -> str:
    """
    Deletes a file from the local filesystem.
    **This is a destructive action and requires user confirmation.**
    Use this tool when you need to remove a file.
    Provide the file path.
    Returns a status message indicating success, failure, or user cancellation.
    """
    file_path = os.path.expanduser(input.file_path) # Expand user home directory (~)
    logging.info(f"Agent called delete_file tool for path: {file_path}")

    # Basic path validation (can be expanded)
    if not os.path.exists(file_path):
        logging.warning(f"Attempted to delete non-existent file: {file_path}")
        return f"Error: File not found at {file_path}"
    if not os.path.isfile(file_path):
         logging.warning(f"Attempted to delete path that is not a file: {file_path}")
         return f"Error: Path {file_path} is not a file."

    console.print(Panel(f"Agent proposes to delete file: [bold red]{file_path}[/bold red]", title="[bold yellow]Action Required: Confirm File Deletion[/bold yellow]", expand=False))

    if Confirm.ask(f"Do you want to confirm deleting {file_path}? This cannot be undone!"):
        try:
            os.remove(file_path)
            logging.info(f"Successfully deleted file: {file_path}")
            console.print(f"[bold green]Successfully deleted {file_path}[/bold green]")
            return f"Successfully deleted file: {file_path}."
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")
            console.print(f"[bold red]Error deleting file {file_path}: {e}[/bold red]")
            return f"Error deleting file {file_path}: {e}"
    else:
        logging.warning(f"User cancelled deleting file: {file_path}")
        console.print(f"[bold yellow]File deletion of {file_path} cancelled by user.[/bold yellow]")
        return f"File deletion of {file_path} cancelled by user."

# 互联网访问工具 (需要 requests 库)
if REQUESTS_AVAILABLE:
    @tool
    def fetch_url_content(input: FetchUrlContentInput) -> str:
        """
        Fetches the content from a given URL using HTTP GET request.
        Use this tool to access information from the internet, such as documentation, APIs, or raw data.
        Provide the URL and an optional timeout in seconds.
        Returns the content of the response (as text), or an error message if the request fails.
        Truncates content if it's too large.
        """
        url = input.url
        timeout = input.timeout
        logging.info(f"Agent called fetch_url_content tool for URL: {url} with timeout {timeout}")

        # Basic URL validation (can be expanded)
        if not url.startswith('http://') and not url.startswith('https://'):
             logging.warning(f"Invalid URL scheme: {url}")
             return f"Error: Invalid URL scheme. Only http:// and https:// are allowed."

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            content = response.text
            logging.info(f"Successfully fetched content from URL: {url}")
            # Truncate content to avoid overwhelming the LLM and console
            max_content_length = 2000 # Limit the content length returned to the LLM
            truncated_content = content[:max_content_length] + '...' if len(content) > max_content_length else content
            logging.debug(f"Fetched content (truncated):\n{truncated_content}")

            return f"Content from {url}:\n```\n{truncated_content}\n```" # Wrap in markdown code block
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching URL {url}: {e}")
            return f"Error fetching URL {url}: {e}"
        except Exception as e:
            logging.error(f"An unexpected error occurred fetching URL {url}: {e}")
            return f"An unexpected error occurred fetching URL {url}: {e}"
else:
    # Placeholder if requests is not available
    @tool
    def fetch_url_content(input: FetchUrlContentInput) -> str:
        """
        Fetches the content from a given URL.
        This tool is currently unavailable because the 'requests' library is not installed.
        """
        return "Internet access tool is currently unavailable. Please install the 'requests' library (`pip install requests`)."


# 系统信息工具
@tool
def get_system_info(input: GetSystemInfoInput = GetSystemInfoInput()) -> str: # Provide default input
    """
    Retrieves basic information about the operating system and system architecture.
    Use this tool to understand the environment the agent is running in.
    Returns system information as a string.
    """
    logging.info(f"Agent called get_system_info tool with detail level: {input.detail_level}")
    info = {
        "System": platform.system(),
        "Node Name": platform.node(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Python Version": platform.python_version(),
        "Current Working Directory": os.getcwd(),
        "User": os.getenv('USER') or os.getenv('USERNAME') # Get current user
    }

    if input.detail_level == "full":
        # Add more detailed info if requested (be mindful of privacy/security)
        try:
            # Example: Disk usage (simplified)
            import shutil
            total, used, free = shutil.disk_usage("/")
            info["Disk Usage (root)"] = {
                "Total": f"{total // (2**30)} GB",
                "Used": f"{used // (2**30)} GB",
                "Free": f"{free // (2**30)} GB"
            }
        except Exception as e:
            logging.warning(f"Could not get disk usage: {e}")
            info["Disk Usage (root)"] = f"Could not retrieve: {e}"

        # Example: Basic CPU count
        try:
            info["CPU Count"] = os.cpu_count()
        except Exception as e:
             logging.warning(f"Could not get CPU count: {e}")
             info["CPU Count"] = f"Could not retrieve: {e}"


    logging.info(f"System info retrieved: {info}")
    # Format as a readable string or JSON
    info_string = "System Information:\n"
    for key, value in info.items():
        if isinstance(value, dict):
            info_string += f"  {key}:\n"
            for sub_key, sub_value in value.items():
                 info_string += f"    {sub_key}: {sub_value}\n"
        else:
            info_string += f"  {key}: {value}\n"

    return info_string


# 将所有工具放在一个列表中
tools = [
    generate_bash_script,
    generate_code,
    read_file,
    write_file,
    delete_file,
    fetch_url_content,
    get_system_info
]

# --- 定义 Agent Prompt ---
# 提示词指导 Agent 的行为和如何使用工具
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a powerful Operations (Ops) Agent. Your goal is to assist the user with various operational tasks on their local system and the internet by using the tools available to you.

Available tools:
{tools}

Use the tools strategically to:
1. Understand the user's request.
2. **If the user asks for a Bash script to perform a task, use the `generate_bash_script` tool.**
3. **If the user asks for code or configuration in a specific language, use the `generate_code` tool.**
4. **If the user asks to read a file, use the `read_file` tool.**
5. **If the user asks to write to or modify a file, use the `write_file` tool.** Remember this requires user confirmation.
6. **If the user asks to delete a file, use the `delete_file` tool.** Remember this requires user confirmation and is destructive.
7. **If the user asks to get information from a URL, use the `fetch_url_content` tool.**
8. **If the user asks about the system environment you are running on, use the `get_system_info` tool.**
9. Combine tool usage if necessary (e.g., generate code, then write it to a file).

**IMPORTANT SAFETY INSTRUCTIONS:**
- **NEVER execute Bash scripts or any code directly.** Your role is to *generate* them or *propose* file operations. The user will handle the execution of scripts *after* you generate them, and you must get user confirmation for file write/delete operations via the tool itself.
- When using `write_file` or `delete_file`, be very clear in your internal thought process about the file path and the intended action. The tool will handle the user confirmation prompt.
- Be cautious with file paths, especially deletion. Double-check the path before proposing deletion.
- If a task seems unsafe, ambiguous, or beyond your capabilities, inform the user instead of attempting it.

Think step-by-step. First, understand the user's goal. Then, determine which tool(s) are needed. If generating code/script, use the appropriate generation tool. If performing file/internet operations, use the relevant tool. Provide clear responses to the user.

{agent_scratchpad}
"""),
    ("human", "{input}"),
])

# --- 辅助函数：解析和执行 Bash 脚本 (带逐条确认) ---

def parse_bash_script(script_content: str) -> List[str]:
    """
    Parses a Bash script string into a list of individual commands.
    Handles comments and empty lines. Basic parsing, may not handle complex syntax perfectly.
    """
    commands = []
    # Remove markdown code block markers if present
    script_content = re.sub(r'```bash\n(.*?)```', r'\1', script_content, flags=re.DOTALL)
    script_content = re.sub(r'```\n(.*?)```', r'\1', script_content, flags=re.DOTALL) # Also handle generic code blocks

    # Simple split by newline, filter comments and empty lines
    for line in script_content.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            # Basic handling for lines ending with \ for continuation (imperfect)
            # If a line ends with \, append the next line (basic approach)
            if commands and commands[-1].endswith('\\'):
                 commands[-1] = commands[-1][:-1] + line # Remove \ and append next line
            else:
                commands.append(line)

    # A more robust parser would be needed for complex scripts with semicolons, &&, ||, etc.
    # For this example, we assume simple commands per line or basic \ continuation.
    # We will execute each *parsed* line as a separate command after confirmation.

    return commands

def execute_script_with_confirmation(script_content: str):
    """
    Parses a Bash script and executes each command sequentially after user confirmation.
    """
    commands = parse_bash_script(script_content)

    if not commands:
        console.print(Panel("[bold yellow]No executable commands found in the script.[/bold yellow]", title="Script Execution", expand=False))
        logging.info("No executable commands found in the script.")
        return

    console.print(Panel("[bold blue]Generated Bash Script:[/bold blue]", title="Script Execution", expand=False))
    console.print(Syntax(script_content, "bash", theme="monokai", line_numbers=True))
    console.print(Rule(characters="="))

    console.print("[bold yellow]Review the script above.[/bold yellow]")
    console.print("[bold yellow]You will be asked to confirm each command before it is executed.[/bold yellow]")
    console.print(Rule(characters="="))

    for i, command in enumerate(commands):
        console.print(f"\n[bold]Command {i+1}/{len(commands)}:[/bold] [blue]{command}[/blue]")

        if Confirm.ask("Execute this command?"):
            logging.info(f"User confirmed execution of command: {command}")
            try:
                # Use shlex.split for safe splitting of command and arguments
                # shell=False is crucial to prevent shell injection
                process = subprocess.run(shlex.split(command), capture_output=True, text=True, check=False) # check=False so it doesn't raise exception on non-zero exit
                console.print(f"[bold]Status:[/bold] {'[green]SUCCESS[/green]' if process.returncode == 0 else '[red]FAILED[/red]'}")
                console.print(f"[bold]Return Code:[/bold] {process.returncode}")
                if process.stdout:
                    console.print("[bold]Stdout:[/bold]")
                    console.print(process.stdout.strip())
                if process.stderr:
                    console.print("[bold]Stderr:[/bold]")
                    console.print(process.stderr.strip(), style="yellow")

                logging.info(f"Command executed. Return Code: {process.returncode}")
                logging.debug(f"Stdout:\n{process.stdout}")
                logging.debug(f"Stderr:\n{process.stderr}")

            except FileNotFoundError:
                console.print(f"[bold red]Error:[/bold red] Command not found. Is it in your PATH?")
                logging.error(f"Command not found: {command}")
            except Exception as e:
                console.print(f"[bold red]An error occurred during execution:[/bold red] {e}")
                logging.error(f"Error executing command {command}: {e}")

        else:
            console.print("[bold yellow]Command skipped by user.[/bold yellow]")
            logging.warning(f"User skipped command: {command}")

    console.print(Rule(characters="="))
    console.print(Panel("[bold green]Script execution finished.[/bold green]", title="Script Execution", expand=False))
    logging.info("Script execution finished.")


# --- 创建 Agent ---

class OpsAgent:
    """
    A LangChain-based agent for operational tasks with file, internet,
    system info, and code generation capabilities, with user confirmation.
    """
    def __init__(self, llm: Any):
        """
        Initializes the OpsAgent.

        Args:
            llm: The LangChain LLM instance to use (e.g., ChatOpenAI, Ollama).
        """
        self.llm = llm
        self.tools = tools # Use the globally defined tools list

        # Pass the LLM instance to tools that need it internally
        # This requires the llm variable to be accessible here, or pass it explicitly
        # For simplicity in this example, we rely on 'llm' being in the global scope
        # when generate_bash_script and generate_code are called.
        # A more robust approach would be to pass llm during tool initialization
        # if tools were classes, or via a factory function.

        # Create Agent
        self.agent = create_structured_chat_agent(self.llm, self.tools, prompt)

        # Create Agent Executor
        # handle_parsing_errors=True helps the agent recover from bad tool outputs
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        logging.info("OpsAgent initialized.")
        console.print(Panel("[bold green]Ops Agent Initialized![/bold green]\nType 'exit' to quit.", title="Status", expand=False))


    def run(self, user_input: str) -> str:
        """
        Runs the Ops Agent on the given user input.

        Args:
            user_input: The task description provided by the user.

        Returns:
            The final output from the Agent (a response, tool output summary, etc.).
        """
        logging.info(f"User input: {user_input}")
        console.print(Rule(characters="="))
        console.print(f"[bold blue]User:[/bold blue] {user_input}")
        console.print(Rule(characters="="))

        try:
            # Use rich.live or rich.status to show Agent is thinking
            # with Status("Agent is thinking...") as status:
            #    status.spinner = "earth" # Example spinner
            #    result = self.agent_executor.invoke({"input": user_input})

            # Or just print verbose output directly
            result = self.agent_executor.invoke({"input": user_input})

            agent_output = result['output']
            logging.info(f"Agent final output:\n{agent_output}")

            # Check if the output contains a Bash script block
            bash_script_match = re.search(r'```bash\n(.*?)```', agent_output, flags=re.DOTALL)
            if bash_script_match:
                script_content = bash_script_match.group(1).strip()
                console.print(Panel("[bold green]Agent generated a Bash script.[/bold green]", title="Agent Output", expand=False))
                # Call the script execution function with confirmation
                execute_script_with_confirmation(script_content)
                # Return a message indicating script was handled
                return "Bash script generated and presented for execution."
            else:
                 # If no script, just print the agent's output
                 console.print(Panel(agent_output, title="Agent Output", expand=False))
                 return agent_output

        except Exception as e:
            logging.error(f"Error running agent: {e}")
            console.print(Panel(f"[bold red]An error occurred while running the agent:[/bold red] {e}", title="Error", expand=False))
            return f"An error occurred: {e}"


# --- 主执行块 (CLI 界面) ---

if __name__ == "__main__":
    # --- 准备 LLM ---
    # **重要：** 根据你使用的 LLM 取消注释并配置
    # 例如使用 OpenAI (需要设置 OPENAI_API_KEY 环境变量)
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # 或其他模型

    # 例如使用 Ollama (确保 Ollama 正在运行并已拉取模型，如 llama3)
    # 注意：较大的模型（如 llama3）通常在遵循指令和使用工具方面表现更好
    llm = None
    try:
        with Status("Connecting to LLM (Ollama)...", spinner="dots") as status:
            llm = Ollama(model="llama3") # 替换为你拉取的模型名称
            # 简单的测试，确保 LLM 可用
            status.update("Testing LLM connection...")
            llm.invoke("Hello, world!")
            status.update("[bold green]LLM connected successfully.[/bold green]")
    except Exception as e:
        console.print(Panel(f"[bold red]错误：无法初始化 LLM。请检查 Ollama 服务是否运行以及模型名称是否正确。错误信息: {e}[/bold red]", title="LLM Error", expand=False))
        llm = None # 标记 LLM 不可用
        sys.exit("LLM initialization failed. Exiting.")


    # --- 检查网络工具可用性 (ping, nmap - 之前安全 Agent 中的检查，这里不直接执行，但可以提示用户) ---
    # 在这个 Ops Agent 中，我们主要依赖 subprocess 执行任意命令 (带确认)
    # 所以不需要像安全 Agent 那样专门检查 ping/nmap，Agent 可以尝试执行任何命令
    # 但用户需要确认。

    # --- 初始化 Agent ---
    ops_agent = OpsAgent(llm=llm)

    # --- CLI 交互循环 ---
    while True:
        console.print(Rule(characters="-"))
        user_input = console.input("[bold green]Enter your operational task (or 'exit' to quit):[/bold green] ")
        console.print(Rule(characters="-"))

        if user_input.lower() == 'exit':
            console.print(Panel("[bold blue]Exiting Ops Agent. Goodbye![/bold blue]", title="Status", expand=False))
            break

        if not user_input.strip():
            console.print("[bold yellow]Please enter a task.[/bold yellow]")
            continue

        # 运行 Agent
        # Agent.run will handle printing the output and executing scripts with confirmation
        ops_agent.run(user_input)

