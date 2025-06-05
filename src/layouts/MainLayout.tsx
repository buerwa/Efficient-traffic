import React, { useState, useEffect } from "react";
import { Layout, Menu, Typography, Avatar, Dropdown, message } from "antd";
import {
  HomeOutlined,
  UserOutlined,
  HistoryOutlined,
  LogoutOutlined,
  BarChartOutlined,
} from "@ant-design/icons";
import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { getUserInfo } from "../api";
import { User } from "../types";

const { Header, Content, Sider } = Layout;
const { Title } = Typography;

const MainLayout: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    fetchUserInfo();
  }, []);

  const fetchUserInfo = async () => {
    try {
      const response = await getUserInfo();
      setUser(response.data);
    } catch (error) {
      console.error("获取用户信息失败", error);
      message.error("获取用户信息失败");
      localStorage.removeItem("token");
      navigate("/login");
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("userRole");
    localStorage.removeItem("username");
    navigate("/login");
    message.success("已退出登录");
  };

  const menuItems = [
    {
      key: "/app",
      icon: <HomeOutlined />,
      label: "交通实况识别",
      onClick: () => navigate("/app"),
    },
    {
      key: "/app/history",
      icon: <HistoryOutlined />,
      label: "历史记录查询",
      onClick: () => navigate("/app/history"),
    },
    {
      key: "/app/analysis",
      icon: <BarChartOutlined />,
      label: "交通数据分析",
      onClick: () => navigate("/app/analysis"),
    },
  ];

  // 管理员菜单项
  if (user?.role === "admin") {
    menuItems.push({
      key: "/app/user-management",
      icon: <UserOutlined />,
      label: "用户管理",
      onClick: () => navigate("/app/user-management"),
    });
  }

  // 用户下拉菜单
  const userMenu = {
    items: [
      {
        key: "1",
        label: "退出登录",
        icon: <LogoutOutlined />,
        onClick: handleLogout,
      },
    ],
  };

  // 确定当前选中的菜单项
  const selectedKey = location.pathname;

  return (
    <Layout style={{ minHeight: "100vh", width: "100%" }}>
      <Header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "#fff",
          padding: "0 24px",
          boxShadow: "0 1px 4px rgba(0,0,0,0.1)",
          width: "100%",
        }}
      >
        <div style={{ display: "flex", alignItems: "center" }}>
          <Title level={3} style={{ margin: 0, color: "#1890ff" }}>
            交通实况识别系统
          </Title>
        </div>
        <Dropdown menu={userMenu} placement="bottomRight">
          <div
            style={{ cursor: "pointer", display: "flex", alignItems: "center" }}
          >
            <Avatar icon={<UserOutlined />} style={{ marginRight: 8 }} />
            <span>{user?.username || "用户"}</span>
          </div>
        </Dropdown>
      </Header>
      <Layout>
        <Sider
          width={200}
          collapsible
          collapsed={collapsed}
          onCollapse={setCollapsed}
          style={{ background: "#fff" }}
        >
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            style={{ height: "100%", borderRight: 0 }}
            items={menuItems}
          />
        </Sider>
        <Layout style={{ padding: "24px", width: "100%" }}>
          <Content
            style={{
              padding: 24,
              margin: 0,
              background: "#fff",
              borderRadius: 4,
              minHeight: 280,
              width: "100%",
            }}
          >
            <Outlet />
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
};

export default MainLayout;
