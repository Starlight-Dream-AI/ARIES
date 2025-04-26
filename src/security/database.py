import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

# 定义数据库文件路径
DATABASE_FILE = "security_incidents.db"

class SecurityIncidentDB:
    """
    SQLite 数据库接口，用于存储网络安全事件。
    """
    def __init__(self, db_path: str = DATABASE_FILE):
        """
        初始化数据库连接并创建表。

        Args:
            db_path: SQLite 数据库文件路径。
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_table()

    def _connect(self):
        """
        建立数据库连接。
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            # 设置 row_factory，使得查询结果可以通过列名访问
            self.conn.row_factory = sqlite3.Row
            print(f"成功连接到数据库: {self.db_path}")
        except sqlite3.Error as e:
            print(f"数据库连接错误: {e}")
            # 在生产环境中，可能需要更复杂的错误处理，例如退出或重试

    def _create_table(self):
        """
        创建存储安全事件的表（如果不存在）。
        """
        sql = """
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            incident_type TEXT NOT NULL,
            source_ip TEXT,
            source_port INTEGER,
            dest_ip TEXT,
            dest_port INTEGER,
            protocol TEXT,
            severity TEXT NOT NULL,
            description TEXT,
            status TEXT NOT NULL DEFAULT 'Open'
        );
        """
        try:
            with self.conn: # 使用 with 语句管理事务和连接关闭
                cursor = self.conn.cursor()
                cursor.execute(sql)
            print("事件表已准备就绪。")
        except sqlite3.Error as e:
            print(f"创建表错误: {e}")

    def add_incident(self,
                     incident_type: str,
                     severity: str,
                     description: str,
                     timestamp: Optional[str] = None,
                     source_ip: Optional[str] = None,
                     source_port: Optional[int] = None,
                     dest_ip: Optional[str] = None,
                     dest_port: Optional[int] = None,
                     protocol: Optional[str] = None,
                     status: str = 'Open') -> Optional[int]:
        """
        向数据库添加一个新的安全事件。

        Args:
            incident_type: 事件类型 (例如: 'Intrusion Attempt', 'Malware Detected', 'Port Scan').
            severity: 严重性 ('Low', 'Medium', 'High', 'Critical').
            description: 事件详细描述。
            timestamp: 事件发生时间 (ISO 8601 格式字符串)，默认为当前时间。
            source_ip: 源 IP 地址。
            source_port: 源端口。
            dest_ip: 目标 IP 地址。
            dest_port: 目标端口。
            protocol: 协议 (例如: 'TCP', 'UDP', 'ICMP').
            status: 事件状态 ('Open', 'Investigating', 'Closed', 'False Positive')，默认为 'Open'。

        Returns:
            新插入记录的 ID，如果插入失败则返回 None。
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat() # 使用 ISO 8601 格式存储时间

        sql = """
        INSERT INTO incidents (timestamp, incident_type, source_ip, source_port,
                               dest_ip, dest_port, protocol, severity, description, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data = (timestamp, incident_type, source_ip, source_port,
                dest_ip, dest_port, protocol, severity, description, status)

        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute(sql, data)
                # with 语句块结束时会自动 commit 或 rollback
                return cursor.lastrowid # 返回最后插入行的 ID
        except sqlite3.Error as e:
            print(f"添加事件错误: {e}")
            return None

    def get_incident_by_id(self, incident_id: int) -> Optional[Dict[str, Any]]:
        """
        根据 ID 查询特定安全事件。

        Args:
            incident_id: 要查询的事件 ID。

        Returns:
            事件记录字典，如果未找到则返回 None。
        """
        sql = "SELECT * FROM incidents WHERE id = ?"
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute(sql, (incident_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row) # 返回字典形式的结果
                else:
                    return None
        except sqlite3.Error as e:
            print(f"查询事件 (ID: {incident_id}) 错误: {e}")
            return None

    def get_all_incidents(self) -> List[Dict[str, Any]]:
        """
        查询所有安全事件。

        Returns:
            事件记录字典列表。
        """
        sql = "SELECT * FROM incidents ORDER BY timestamp DESC" # 按时间倒序排列
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [dict(row) for row in rows] # 返回字典列表
        except sqlite3.Error as e:
            print(f"查询所有事件错误: {e}")
            return []

    def get_incidents_by_type(self, incident_type: str) -> List[Dict[str, Any]]:
        """
        根据事件类型查询安全事件。

        Args:
            incident_type: 要查询的事件类型。

        Returns:
            事件记录字典列表。
        """
        sql = "SELECT * FROM incidents WHERE incident_type = ? ORDER BY timestamp DESC"
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute(sql, (incident_type,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"按类型查询事件错误 ({incident_type}): {e}")
            return []

    def get_incidents_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """
        根据严重性查询安全事件。

        Args:
            severity: 要查询的严重性。

        Returns:
            事件记录字典列表。
        """
        sql = "SELECT * FROM incidents WHERE severity = ? ORDER BY timestamp DESC"
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute(sql, (severity,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"按严重性查询事件错误 ({severity}): {e}")
            return []

    def get_incidents_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        根据状态查询安全事件。

        Args:
            status: 要查询的状态。

        Returns:
            事件记录字典列表。
        """
        sql = "SELECT * FROM incidents WHERE status = ? ORDER BY timestamp DESC"
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute(sql, (status,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"按状态查询事件错误 ({status}): {e}")
            return []

    def update_incident_status(self, incident_id: int, new_status: str) -> int:
        """
        更新特定安全事件的状态。

        Args:
            incident_id: 要更新的事件 ID。
            new_status: 新的状态。

        Returns:
            更新的行数。
        """
        sql = "UPDATE incidents SET status = ? WHERE id = ?"
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute(sql, (new_status, incident_id))
                # with 语句块结束时会自动 commit 或 rollback
                return cursor.rowcount # 返回影响的行数
        except sqlite3.Error as e:
            print(f"更新事件状态 (ID: {incident_id}) 错误: {e}")
            return 0

    def delete_incident(self, incident_id: int) -> int:
        """
        根据 ID 删除特定安全事件。

        Args:
            incident_id: 要删除的事件 ID。

        Returns:
            删除的行数。
        """
        sql = "DELETE FROM incidents WHERE id = ?"
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute(sql, (incident_id,))
                # with 语句块结束时会自动 commit 或 rollback
                return cursor.rowcount # 返回影响的行数
        except sqlite3.Error as e:
            print(f"删除事件 (ID: {incident_id}) 错误: {e}")
            return 0

    def close(self):
        """
        关闭数据库连接。
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            print("数据库连接已关闭。")

    def __del__(self):
        """
        在对象销毁时确保关闭连接。
        """
        self.close()

# --- 示例用法 ---
if __name__ == "__main__":
    # 创建数据库接口实例
    db = SecurityIncidentDB()

    # 添加一些安全事件
    print("\n添加事件...")
    incident_id_1 = db.add_incident(
        incident_type='Port Scan',
        severity='Low',
        description='Detected Nmap scan from external IP.',
        source_ip='203.0.113.10',
        dest_ip='192.168.1.100',
        protocol='TCP'
    )
    print(f"添加事件 ID: {incident_id_1}")

    incident_id_2 = db.add_incident(
        incident_type='Malware Detected',
        severity='High',
        description='Antivirus detected a known malware signature.',
        source_ip='192.168.1.50',
        protocol='N/A'
    )
    print(f"添加事件 ID: {incident_id_2}")

    incident_id_3 = db.add_incident(
        incident_type='Intrusion Attempt',
        severity='Critical',
        description='Failed login attempts on SSH from multiple IPs.',
        dest_ip='your_server_ip',
        dest_port=22,
        protocol='TCP'
    )
    print(f"添加事件 ID: {incident_id_3}")

    # 查询所有事件
    print("\n所有事件:")
    all_incidents = db.get_all_incidents()
    for inc in all_incidents:
        print(inc)

    # 根据 ID 查询特定事件
    if incident_id_1 is not None:
        print(f"\n查询 ID 为 {incident_id_1} 的事件:")
        incident = db.get_incident_by_id(incident_id_1)
        print(incident)

    # 根据类型查询事件
    print("\n查询类型为 'Malware Detected' 的事件:")
    malware_incidents = db.get_incidents_by_type('Malware Detected')
    for inc in malware_incidents:
        print(inc)

    # 根据严重性查询事件
    print("\n查询严重性为 'Critical' 的事件:")
    critical_incidents = db.get_incidents_by_severity('Critical')
    for inc in critical_incidents:
        print(inc)

    # 更新事件状态
    if incident_id_1 is not None:
        print(f"\n更新事件 ID {incident_id_1} 的状态为 'Closed'...")
        updated_count = db.update_incident_status(incident_id_1, 'Closed')
        print(f"更新了 {updated_count} 条记录。")
        print(f"再次查询 ID 为 {incident_id_1} 的事件:")
        incident = db.get_incident_by_id(incident_id_1)
        print(incident)

    # 删除事件
    if incident_id_2 is not None:
        print(f"\n删除事件 ID {incident_id_2}...")
        deleted_count = db.delete_incident(incident_id_2)
        print(f"删除了 {deleted_count} 条记录。")
        print(f"再次查询 ID 为 {incident_id_2} 的事件:")
        incident = db.get_incident_by_id(incident_id_2)
        print(incident) # 应该返回 None

    # 再次查询所有事件，确认删除和更新
    print("\n更新和删除后的所有事件:")
    all_incidents_after = db.get_all_incidents()
    for inc in all_incidents_after:
        print(inc)

    # 关闭数据库连接 (虽然 with 语句和 __del__ 会自动处理，但显式调用也是好的习惯)
    db.close()
