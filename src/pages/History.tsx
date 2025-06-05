import React, { useEffect, useState } from "react";
import {
  Card,
  Table,
  Button,
  Space,
  message,
  Select,
  Image,
  Popconfirm,
  Tag,
  Row,
  Col,
  Empty,
} from "antd";
import { DeleteOutlined, SearchOutlined } from "@ant-design/icons";
import { getPredictionHistory, deletePrediction } from "../api";
import moment from "moment";
import type { PredictionHistoryItem } from "../types";

const { Option } = Select;

const typeOptions = [
  { value: "all", label: "全部" },
  { value: "交通顺畅", label: "交通顺畅" },
  { value: "交通拥堵", label: "交通拥堵" },
  { value: "交通事故", label: "交通事故" },
  { value: "车辆起火", label: "车辆起火" },
];

const History: React.FC = () => {
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedType, setSelectedType] = useState<string>("all");

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const predictionType = selectedType !== "all" ? selectedType : undefined;

      console.log("开始获取历史记录，参数:", {
        limit: 1000,
        predictionType,
      });

      const response = await getPredictionHistory({
        limit: 1000,
        predictionType,
        timestamp: Date.now(), // 添加时间戳防止缓存
      });

      console.log("历史记录API响应:", response);
      console.log("响应数据结构:", response.data);

      // 提取预测记录，兼容两种可能的API响应格式
      const records = response.data.items || response.data.predictions || [];
      const total = response.data.total || response.data.count || 0;

      console.log(`获取到 ${records.length} 条历史记录，总计: ${total}`);

      if (records.length > 0) {
        console.log("第一条记录示例:", records[0]);
      }

      // 处理记录，确保数据格式正确
      const processedItems = records.map((item: PredictionHistoryItem) => {
        // 确保每个条目都有_id字段
        if (!item._id && (item as PredictionHistoryItem & { id?: string }).id) {
          item._id = (item as PredictionHistoryItem & { id?: string }).id!;
        }

        // 处理prediction字段，它可能是字符串或对象
        if (typeof item.prediction === "object") {
          // 如果是对象，提取预测结果
          const predObj = item.prediction as {
            class_name?: string;
            prediction?: string;
          };
          if (predObj.class_name) {
            item.prediction = predObj.class_name;
          } else if (predObj.prediction) {
            item.prediction = predObj.prediction;
          }
        }

        return item;
      });

      // 根据时间戳或创建时间进行倒序排序，确保最新的记录显示在最前面
      const sortedItems = processedItems.sort(
        (a: PredictionHistoryItem, b: PredictionHistoryItem) => {
          const timeA = a.timestamp || a.created_at || "";
          const timeB = b.timestamp || b.created_at || "";
          return new Date(timeB).getTime() - new Date(timeA).getTime(); // 降序排列
        }
      );

      console.log(
        "排序后第一条记录:",
        sortedItems.length > 0 ? sortedItems[0] : "无数据"
      );
      setHistory(sortedItems);
    } catch (error) {
      console.error("Failed to fetch history:", error);
      message.error("获取历史记录失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleSearch = () => {
    fetchHistory();
  };

  const handleDelete = async (id: string) => {
    try {
      await deletePrediction(id);
      message.success("删除成功");
      fetchHistory();
    } catch (error) {
      console.error("Failed to delete prediction:", error);
      message.error("删除失败");
    }
  };

  const columns = [
    {
      title: "图片",
      dataIndex: "image_url",
      key: "image",
      render: (url: string) => {
        // 检查URL是否为空或undefined
        if (!url) {
          return <div>暂无图片</div>;
        }

        // 使用固定的后端URL格式访问
        const imageUrl = url.startsWith("http")
          ? url
          : `http://localhost:8000${url.startsWith("/") ? url : `/${url}`}`;

        console.log("尝试加载图片:", imageUrl);

        return <Image width={100} src={imageUrl} />;
      },
    },
    {
      title: "预测结果",
      dataIndex: "prediction",
      key: "prediction",
      render: (prediction: string) => {
        // 使用预定义的颜色映射
        const colors: Record<string, string> = {
          交通顺畅: "success",
          sparse_traffic: "success",
          交通拥堵: "warning",
          dense_traffic: "warning",
          交通事故: "error",
          accident: "error",
          车辆起火: "volcano",
          fire: "volcano",
        };
        return <Tag color={colors[prediction] || "blue"}>{prediction}</Tag>;
      },
    },
    {
      title: "置信度",
      dataIndex: "confidence",
      key: "confidence",
      render: (confidence: number) => {
        // 显示为百分比格式
        return confidence ? `${(confidence * 100).toFixed(2)}%` : "未知";
      },
    },
    {
      title: "时间",
      dataIndex: "timestamp",
      key: "timestamp",
      render: (timestamp: string, record: PredictionHistoryItem) => {
        const timeStr = timestamp || record.created_at || "";
        if (timeStr) {
          // 匹配年-月-日 时:分:秒 格式
          const match = timeStr.match(
            /(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})|(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})/
          );
          if (match) {
            return match[0].replace("T", " "); // 将ISO格式的T替换为空格
          }
          return moment(timeStr).format("YYYY-MM-DD HH:mm:ss");
        }
        return timeStr;
      },
    },
    {
      title: "操作",
      key: "action",
      render: (_: unknown, record: PredictionHistoryItem) => (
        <Popconfirm
          title="确定要删除这条记录吗?"
          onConfirm={() => handleDelete(record._id)}
          okText="是"
          cancelText="否"
        >
          <Button type="link" danger icon={<DeleteOutlined />}>
            删除
          </Button>
        </Popconfirm>
      ),
    },
  ];

  return (
    <Card title="历史预测记录" style={{ marginBottom: 20 }}>
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={8} md={8} lg={6}>
          <Select
            value={selectedType}
            onChange={setSelectedType}
            style={{ width: "100%" }}
            placeholder="选择标志类型"
          >
            {typeOptions.map((option) => (
              <Option key={option.value} value={option.value}>
                {option.label}
              </Option>
            ))}
          </Select>
        </Col>
        <Col xs={24} sm={8} md={8} lg={6}>
          <Space>
            <Button
              type="primary"
              icon={<SearchOutlined />}
              onClick={handleSearch}
            >
              搜索
            </Button>
          </Space>
        </Col>
      </Row>

      {history.length > 0 ? (
        <Table
          columns={columns}
          dataSource={history}
          rowKey="_id"
          loading={loading}
          pagination={false}
          scroll={{ x: "max-content", y: 500 }}
        />
      ) : (
        <div style={{ padding: "50px 0" }}>
          <Empty
            description="暂无预测记录"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        </div>
      )}
    </Card>
  );
};

export default History;
