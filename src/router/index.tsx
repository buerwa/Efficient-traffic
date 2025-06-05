import React from "react";
import { createBrowserRouter, Navigate } from "react-router-dom";
import Login from "../pages/Login";
import MainLayout from "../layouts/MainLayout";
import ImageRecognition from "../pages/ImageRecognition";
import UserManagement from "../pages/UserManagement";
import History from "../pages/History";
import TrafficAnalysis from "../pages/TrafficAnalysis";

// 权限验证HOC
const RequireAuth = ({ children }: { children: React.ReactElement }) => {
  const isAuthenticated = localStorage.getItem("token");
  return isAuthenticated ? children : <Navigate to="/login" replace />;
};

// 管理员权限验证HOC
const RequireAdmin = ({ children }: { children: React.ReactElement }) => {
  const isAuthenticated = localStorage.getItem("token");
  const userRole = localStorage.getItem("userRole");
  return isAuthenticated && userRole === "admin" ? (
    children
  ) : isAuthenticated ? (
    <Navigate to="/app" replace />
  ) : (
    <Navigate to="/login" replace />
  );
};

const router = createBrowserRouter([
  {
    path: "/",
    element: <Navigate to="/app" replace />,
  },
  {
    path: "/login",
    element: <Login />,
  },
  {
    path: "/app",
    element: (
      <RequireAuth>
        <MainLayout />
      </RequireAuth>
    ),
    children: [
      {
        path: "",
        element: <ImageRecognition />,
      },
      {
        path: "user-management",
        element: (
          <RequireAdmin>
            <UserManagement />
          </RequireAdmin>
        ),
      },
      {
        path: "history",
        element: <History />,
      },
      {
        path: "analysis",
        element: <TrafficAnalysis />,
      },
    ],
  },
  // 捕获所有未匹配的路径并重定向到登录
  {
    path: "*",
    element: <Navigate to="/login" replace />,
  },
]);

export default router;
