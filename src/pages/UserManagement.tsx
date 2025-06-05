import React, { useState, useEffect } from "react";
import {
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  message,
  Typography,
  Space,
  Popconfirm,
  Card,
  Row,
  Col,
} from "antd";
import {
  UserAddOutlined,
  EditOutlined,
  DeleteOutlined,
} from "@ant-design/icons";
import { createUser, getUsers, updateUser, deleteUser } from "../api";
import { User, UserUpdate } from "../types";

const { Title } = Typography;
const { Option } = Select;

const UserManagement: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [modalTitle, setModalTitle] = useState("");
  const [form] = Form.useForm();
  const [editingUser, setEditingUser] = useState<string | null>(null);
  const [search, setSearch] = useState<string>("");

  // 获取所有用户
  const fetchUsers = async () => {
    setLoading(true);
    try {
      const response = await getUsers(search);
      setUsers(response.data);
    } catch (error) {
      message.error("获取用户列表失败");
      console.error("获取用户列表失败:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  // 打开创建用户模态框
  const showCreateModal = () => {
    form.resetFields();
    setModalTitle("创建新用户");
    setEditingUser(null);
    setModalVisible(true);
  };

  // 打开编辑用户模态框
  const showEditModal = (user: User) => {
    form.setFieldsValue({
      username: user.username,
      fullName: user.full_name,
      role: user.role,
      isActive: user.is_active,
    });
    setModalTitle("编辑用户");
    setEditingUser(user.username);
    setModalVisible(true);
  };

  // 提交表单
  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();

      if (editingUser) {
        // 更新用户
        const userData: UserUpdate = {
          full_name: values.fullName,
          role: values.role,
          is_active: values.isActive,
        };

        if (values.password) {
          userData.password = values.password;
        }

        await updateUser(editingUser, userData);
        message.success("用户更新成功");
      } else {
        // 创建用户
        try {
          await createUser(values.username, values.password, values.fullName);
          message.success("用户创建成功");
        } catch (error: unknown) {
          console.error("创建用户失败", error);
          // 显示详细错误信息
          if (error && typeof error === "object" && "response" in error) {
            const axiosError = error as {
              response: {
                status: number;
                statusText: string;
                data: Record<string, unknown>;
              };
              message: string;
            };
            message.error(
              `创建用户失败: ${axiosError.response.status} ${
                axiosError.response.statusText
              } - ${JSON.stringify(axiosError.response.data)}`
            );
          } else if (error instanceof Error) {
            message.error(`创建用户失败: ${error.message}`);
          } else {
            message.error("创建用户失败，未知错误");
          }
          return; // 创建失败时不关闭模态框
        }
      }

      setModalVisible(false);
      fetchUsers(); // 刷新用户列表
    } catch (error) {
      console.error("操作失败", error);
      message.error("操作失败");
    }
  };

  // 删除用户
  const handleDelete = async (username: string) => {
    try {
      await deleteUser(username);
      message.success("用户删除成功");
      fetchUsers(); // 刷新用户列表
    } catch (error) {
      console.error("删除用户失败", error);
      message.error("删除用户失败");
    }
  };

  // 表格列定义
  const columns = [
    {
      title: "用户名",
      dataIndex: "username",
      key: "username",
    },
    {
      title: "姓名",
      dataIndex: "full_name",
      key: "full_name",
      render: (text: string) => text || "-",
    },
    {
      title: "角色",
      dataIndex: "role",
      key: "role",
      render: (role: string) => (role === "admin" ? "管理员" : "普通用户"),
    },
    {
      title: "状态",
      dataIndex: "is_active",
      key: "is_active",
      render: (active: boolean) => (
        <span style={{ color: active ? "#52c41a" : "#f5222d" }}>
          {active ? "活跃" : "禁用"}
        </span>
      ),
    },
    {
      title: "操作",
      key: "action",
      render: (_: unknown, record: User) => (
        <Space size="small">
          <Button
            type="text"
            icon={<EditOutlined />}
            onClick={() => showEditModal(record)}
          />
          <Popconfirm
            title="确定删除此用户吗？"
            onConfirm={() => handleDelete(record.username)}
            okText="确定"
            cancelText="取消"
            disabled={record.username === localStorage.getItem("username")}
          >
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
              disabled={record.username === localStorage.getItem("username")}
            />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ width: "100%" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
          width: "100%",
        }}
      >
        <Title level={2}>交通实况系统用户管理</Title>
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={8}>
            <Input.Search
              placeholder="搜索用户名或全名"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onSearch={fetchUsers}
              style={{ width: "100%" }}
            />
          </Col>
          <Col span={16} style={{ textAlign: "right" }}>
            <Button
              type="primary"
              icon={<UserAddOutlined />}
              onClick={showCreateModal}
            >
              新增用户
            </Button>
          </Col>
        </Row>
      </div>

      <Card style={{ width: "100%" }}>
        <Table
          columns={columns}
          dataSource={users}
          rowKey="username"
          loading={loading}
          pagination={{ pageSize: 10 }}
          style={{ width: "100%" }}
          scroll={{ x: "max-content", y: 500 }}
        />
      </Card>

      <Modal
        title={modalTitle}
        open={modalVisible}
        onOk={handleSubmit}
        onCancel={() => setModalVisible(false)}
        okText="提交"
        cancelText="取消"
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="username"
            label="用户名"
            rules={[
              { required: true, message: "请输入用户名" },
              { min: 3, message: "用户名至少3个字符" },
            ]}
            hidden={!!editingUser}
          >
            <Input placeholder="请输入用户名" />
          </Form.Item>

          <Form.Item
            name="password"
            label="密码"
            rules={[
              { required: !editingUser, message: "请输入密码" },
              { min: 4, message: "密码至少4个字符" },
            ]}
            extra={editingUser ? "留空表示不修改密码" : ""}
          >
            <Input.Password placeholder="请输入密码" />
          </Form.Item>

          <Form.Item name="fullName" label="姓名">
            <Input placeholder="请输入姓名" />
          </Form.Item>

          <Form.Item
            name="role"
            label="角色"
            initialValue="user"
            rules={[{ required: true, message: "请选择角色" }]}
          >
            <Select placeholder="请选择角色">
              <Option value="user">普通用户</Option>
              <Option value="admin">管理员</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="isActive"
            label="状态"
            valuePropName="checked"
            initialValue={true}
          >
            <Switch checkedChildren="活跃" unCheckedChildren="禁用" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default UserManagement;
