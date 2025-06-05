import React, { useState } from "react";
import { Form, Input, Button, Card, Typography, message, Alert } from "antd";
import { UserOutlined, LockOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { login, getUserInfo } from "../api";
import { LoginForm } from "../types";

const { Title } = Typography;

const Login: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const onFinish = async (values: LoginForm) => {
    try {
      setLoading(true);
      setError(null);

      const response = await login(values.username, values.password);
      localStorage.setItem("token", response.data.access_token);

      // 获取用户信息
      const userInfo = await getUserInfo();
      localStorage.setItem("userRole", userInfo.data.role);
      localStorage.setItem("username", userInfo.data.username);

      message.success("登录成功！");
      navigate("/app");
    } catch (err) {
      setError("用户名或密码错误，请重试");
      console.error("登录失败:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        background: "#f0f2f5",
        margin: "0 auto",
        maxWidth: "100vw",
      }}
    >
      <Card style={{ width: 400, boxShadow: "0 4px 8px rgba(0,0,0,0.1)" }}>
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <Title level={2}>交通实况识别系统</Title>
          <Title level={4}>用户登录</Title>
        </div>

        {error && (
          <Alert
            message={error}
            type="error"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        <Form
          name="login"
          initialValues={{ remember: true }}
          onFinish={onFinish}
          size="large"
        >
          <Form.Item
            name="username"
            rules={[{ required: true, message: "请输入用户名" }]}
          >
            <Input prefix={<UserOutlined />} placeholder="用户名" />
          </Form.Item>

          <Form.Item
            name="password"
            rules={[{ required: true, message: "请输入密码" }]}
          >
            <Input.Password prefix={<LockOutlined />} placeholder="密码" />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading} block>
              登录
            </Button>
          </Form.Item>

          <div style={{ textAlign: "center" }}>
            <p>默认管理员: root / 1234</p>
          </div>
        </Form>
      </Card>
    </div>
  );
};

export default Login;
