import React, { useState, useEffect, useMemo } from "react";
import {
  Card,
  Row,
  Col,
  Spin,
  Typography,
  Statistic,
  Divider,
  Select,
  message,
  Space,
  Empty,
  Button,
  Tooltip,
  Tag,
} from "antd";
import { Pie, Line } from "@ant-design/plots";
import type { LineConfig } from "@ant-design/plots";
import { getPredictionHistory } from "../api";
import { PredictionHistoryItem } from "../types";
import {
  PieChartOutlined,
  LineChartOutlined,
  CarOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  FireOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  HistoryOutlined,
} from "@ant-design/icons";

const { Title } = Typography;
const { Option } = Select;

const styles = {
  chartContainer: {
    height: "320px",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
};

const TrafficAnalysis: React.FC = () => {
  const [historyData, setHistoryData] = useState<PredictionHistoryItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [recordCount, setRecordCount] = useState<number>(50);

  useEffect(() => {
    fetchHistoryData();
  }, [recordCount]);

  // 获取历史记录数据
  const fetchHistoryData = async () => {
    try {
      setLoading(true);
      const response = await getPredictionHistory({
        limit: recordCount * 2, // 获取更多记录以确保有足够的数据
        timestamp: Date.now(), // 添加时间戳防止缓存
      });

      console.log("历史记录API响应:", response);

      // 提取预测记录
      const records = response.data.items || response.data.predictions || [];

      // 处理记录，确保数据格式正确
      const processedItems = records.map((item: PredictionHistoryItem) => {
        // 确保每个条目都有_id字段
        if (!item._id && (item as unknown as { id?: string }).id) {
          item._id = (item as unknown as { id: string }).id;
        }

        // 处理prediction字段，它可能是字符串或对象
        if (typeof item.prediction === "object") {
          const predObj = item.prediction as unknown as {
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

      // 只保留所需数量的记录
      const limitedItems = sortedItems.slice(0, recordCount);

      setHistoryData(limitedItems);
      console.log(`已获取 ${limitedItems.length} 条历史记录用于数据分析`);
    } catch (error) {
      console.error("获取历史记录失败:", error);
      message.error("获取历史记录失败");
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    setLoading(true);
    fetchHistoryData().finally(() => {
      setLoading(false);
      message.success("数据已刷新");
    });
  };

  // Mock pie data (仅作为无数据时的备用)
  const mockPieData = useMemo(
    () => [
      { type: "交通顺畅", value: 28 },
      { type: "交通拥堵", value: 12 },
      { type: "交通事故", value: 3 },
      { type: "车辆起火", value: 1 },
    ],
    []
  );

  // 从历史记录生成饼图数据
  const pieData = useMemo(() => {
    if (!historyData || historyData.length === 0) {
      console.log("没有历史记录数据，使用模拟数据");
      return mockPieData;
    }

    console.log("使用历史记录数据生成饼图");

    // 统计各类型的数量
    const typeCounts: Record<string, number> = {};

    historyData.forEach((item: PredictionHistoryItem) => {
      const prediction = item.prediction || "";

      // 对不同的预测类型进行归类和标准化
      let standardType = prediction;

      // 标准化英文类型到中文
      if (prediction === "sparse_traffic") standardType = "交通顺畅";
      else if (
        prediction === "dense_traffic" ||
        prediction === "congested_traffic"
      )
        standardType = "交通拥堵";
      else if (prediction === "accident") standardType = "交通事故";
      else if (prediction === "fire") standardType = "车辆起火";

      // 累加计数
      typeCounts[standardType] = (typeCounts[standardType] || 0) + 1;
    });

    console.log("历史记录类型统计:", typeCounts);

    // 转换为饼图数据格式
    const result = Object.keys(typeCounts).map((type) => ({
      type,
      value: typeCounts[type],
    }));

    console.log("生成的历史记录饼图数据:", result);
    return result.length > 0 ? result : mockPieData;
  }, [historyData, mockPieData]);

  // 计算总数
  const totalCount = useMemo(() => {
    return pieData.reduce(
      (sum: number, item: { type: string; value: number }) => sum + item.value,
      0
    );
  }, [pieData]);

  // 从历史记录计算平均置信度
  const avgConfidence = useMemo(() => {
    if (!historyData || historyData.length === 0) return 0.9;

    const validConfidences = historyData
      .map((item) => item.confidence)
      .filter(
        (confidence): confidence is number =>
          typeof confidence === "number" && !isNaN(confidence)
      );

    if (validConfidences.length === 0) return 0.9;

    const sum = validConfidences.reduce((total, value) => total + value, 0);
    return sum / validConfidences.length;
  }, [historyData]);

  // 从历史记录获取置信度趋势数据
  const confidenceTrendData = useMemo(() => {
    if (!historyData || historyData.length === 0) {
      return [];
    }

    // 获取足够的记录
    const records = historyData.slice(0, recordCount);

    // 反转记录顺序，让最旧的记录在左侧，最新的在右侧
    const trendRecords = [...records].reverse();

    // 为每条记录创建趋势数据点
    return trendRecords.map((item, index) => {
      const timeStr = item.timestamp || item.created_at || "";
      let date = "";

      if (timeStr) {
        // 提取日期部分
        const match = timeStr.match(/(\d{4}-\d{2}-\d{2})/);
        if (match) {
          date = match[0];
        }
      }

      return {
        index: index + 1, // 从1开始的索引
        record_number: index + 1,
        date: date,
        prediction: item.prediction,
        avg_confidence: item.confidence || 0,
        image_url: item.image_url,
      };
    });
  }, [historyData, recordCount]);

  // 判断是否显示饼图
  const showPieChart = useMemo(() => {
    return pieData.length > 0;
  }, [pieData]);

  // 判断是否显示折线图
  const showLineChart = useMemo(() => {
    return confidenceTrendData.length > 0;
  }, [confidenceTrendData]);

  // 饼图配置
  const pieConfig = useMemo(() => {
    // 创建颜色映射
    const getColorByType = (type: string): string => {
      if (type === "交通顺畅" || type === "sparse_traffic") {
        return "#52c41a"; // 绿色
      } else if (
        type === "车辆拥堵" ||
        type === "交通拥堵" ||
        type === "dense_traffic" ||
        type === "congested_traffic"
      ) {
        return "#faad14"; // 黄色
      } else if (type === "交通事故" || type === "accident") {
        return "#f5222d"; // 红色
      } else if (type === "车辆起火" || type === "fire") {
        return "#ff4d4f"; // 橙红色
      } else {
        return "#1890ff"; // 蓝色(默认)
      }
    };

    // 为每种类型指定固定颜色
    const colors: string[] = pieData.map((item) => getColorByType(item.type));

    return {
      appendPadding: 10,
      data: pieData,
      angleField: "value",
      colorField: "type",
      radius: 1,
      innerRadius: 0.6,
      // 使用兼容的颜色数组形式
      color: colors,
      label: {
        type: "inner",
        offset: "-50%",
        content: "{value}",
        style: {
          textAlign: "center",
          fontSize: 14,
        },
      },
      interactions: [{ type: "element-selected" }, { type: "element-active" }],
      statistic: {
        title: {
          style: {
            whiteSpace: "pre-wrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            fontSize: "16px",
          },
          content: "交通状况",
        },
        content: {
          style: {
            whiteSpace: "pre-wrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            fontSize: "24px",
          },
          content: totalCount.toString(),
        },
      },
    };
  }, [pieData, totalCount]);

  // 折线图配置
  const lineConfig = useMemo(() => {
    return {
      data: confidenceTrendData,
      padding: "auto" as const,
      xField: "index",
      yField: "avg_confidence",
      xAxis: {
        title: {
          text: "历史记录序号 (从左到右为旧到新)",
        },
        label: {
          formatter: (text: string) => text,
        },
      },
      yAxis: {
        label: {
          formatter: (val: string) => `${(parseFloat(val) * 100).toFixed(1)}%`,
        },
        min: 0.7,
        max: 1,
      },
      tooltip: {
        customContent: (
          title: string,
          items: Array<{ value: string | number }>
        ) => {
          if (items.length > 0) {
            const { value } = items[0];
            const record = confidenceTrendData.find(
              (item) => item.index?.toString() === title
            );
            return `<div style="padding: 8px;">
              <div>记录: 第${title}条</div>
              ${record?.date ? `<div>日期: ${record.date}</div>` : ""}
              ${
                record?.prediction
                  ? `<div>预测: ${record.prediction}</div>`
                  : ""
              }
              <div>置信度: ${(Number(value) * 100).toFixed(2)}%</div>
            </div>`;
          }
          return "";
        },
      },
      point: {
        size: 5,
        shape: "diamond",
      },
      smooth: true,
      theme: {
        styleSheet: {
          brandColor: "#1890ff",
        },
      },
    } as LineConfig;
  }, [confidenceTrendData]);

  // 图标映射
  const typeIcons = {
    交通顺畅: <CheckCircleOutlined style={{ color: "#52c41a" }} />,
    交通拥堵: <WarningOutlined style={{ color: "#faad14" }} />,
    车辆拥堵: <WarningOutlined style={{ color: "#faad14" }} />,
    交通事故: <CarOutlined style={{ color: "#f5222d" }} />,
    车辆起火: <FireOutlined style={{ color: "#ff4d4f" }} />,
    // 增加英文映射以防显示英文类型
    sparse_traffic: <CheckCircleOutlined style={{ color: "#52c41a" }} />,
    dense_traffic: <WarningOutlined style={{ color: "#faad14" }} />,
    congested_traffic: <WarningOutlined style={{ color: "#faad14" }} />,
    accident: <CarOutlined style={{ color: "#f5222d" }} />,
    fire: <FireOutlined style={{ color: "#ff4d4f" }} />,
  };

  // 交通事故率计算
  const accidentRate = useMemo(() => {
    // 找出事故和起火的记录数
    const accidentValue =
      pieData.find((item) => item.type === "交通事故")?.value || 0;
    const fireValue =
      pieData.find((item) => item.type === "车辆起火")?.value || 0;
    // 总记录数
    const total = totalCount;

    return total > 0 ? ((accidentValue + fireValue) / total) * 100 : 0;
  }, [pieData, totalCount]);

  // 交通顺畅率计算
  const smoothTrafficRate = useMemo(() => {
    // 找出顺畅交通的记录数
    const smoothValue =
      pieData.find((item) => item.type === "交通顺畅")?.value || 0;
    // 总记录数
    const total = totalCount;

    return total > 0 ? (smoothValue / total) * 100 : 0;
  }, [pieData, totalCount]);

  if (loading) {
    return (
      <Spin
        size="large"
        tip="加载数据中..."
        style={{
          display: "flex",
          justifyContent: "center",
          marginTop: "100px",
        }}
      />
    );
  }

  return (
    <div
      style={{
        padding: "20px",
        height: "100%",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "20px",
        }}
      >
        <Title level={2}>交通数据分析</Title>
        <Space>
          <Button
            type="primary"
            icon={<ReloadOutlined />}
            onClick={handleRefresh}
          >
            刷新数据
          </Button>
        </Space>
      </div>

      {/* 使用Flex布局并添加可滚动区域 */}
      <div style={{ flex: 1, overflow: "auto", paddingRight: "4px" }}>
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={6} lg={6}>
            <Card hoverable>
              <Statistic
                title="识别总数"
                value={totalCount}
                suffix="次"
                valueStyle={{ color: "#1890ff" }}
                prefix={<CarOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6} lg={6}>
            <Card hoverable>
              <Statistic
                title="平均置信度"
                value={avgConfidence * 100}
                precision={2}
                suffix="%"
                valueStyle={{ color: "#52c41a" }}
                prefix={<LineChartOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6} lg={6}>
            <Card hoverable>
              <Statistic
                title="交通事故率"
                value={accidentRate}
                precision={2}
                suffix="%"
                valueStyle={{ color: "#f5222d" }}
                prefix={<WarningOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6} lg={6}>
            <Card hoverable>
              <Statistic
                title="交通顺畅率"
                value={smoothTrafficRate}
                precision={2}
                suffix="%"
                valueStyle={{ color: "#52c41a" }}
                prefix={<CheckCircleOutlined />}
              />
            </Card>
          </Col>
        </Row>

        <Divider />

        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <Card
              title={
                <Space>
                  <PieChartOutlined />
                  <span>交通实况识别类型分布</span>
                </Space>
              }
              hoverable
              style={{ height: "100%" }}
              extra={
                <Tooltip title="基于历史预测记录">
                  <Tag color="#108ee9">
                    <HistoryOutlined />
                    历史记录
                  </Tag>
                </Tooltip>
              }
            >
              {showPieChart ? (
                <div style={styles.chartContainer}>
                  <Pie
                    {...pieConfig}
                    legend={{
                      position: "bottom",
                      layout: "horizontal",
                      itemName: {
                        style: {
                          fontSize: 14,
                        },
                      },
                    }}
                    animation={{
                      appear: {
                        animation: "fade-in",
                        duration: 1500,
                      },
                    }}
                  />
                </div>
              ) : (
                <Empty
                  description="暂无数据"
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  style={{ margin: "50px 0" }}
                />
              )}
              {showPieChart && (
                <div style={{ marginTop: "20px", padding: "0 20px" }}>
                  <Row gutter={[16, 16]}>
                    {pieData.map((item) => (
                      <Col span={12} key={item.type}>
                        <Card
                          size="small"
                          hoverable
                          style={{
                            borderRadius: "8px",
                            transition: "all 0.3s",
                            boxShadow: "0 2px 8px rgba(0, 0, 0, 0.09)",
                          }}
                          bodyStyle={{ padding: "12px" }}
                        >
                          <Space align="center">
                            {typeIcons[item.type as keyof typeof typeIcons] || (
                              <InfoCircleOutlined
                                style={{ color: "#1890ff" }}
                              />
                            )}
                            <div>
                              <div style={{ fontWeight: "bold" }}>
                                {item.type}
                              </div>
                              <div
                                style={{ fontSize: "18px", color: "#1890ff" }}
                              >
                                {item.value}次
                              </div>
                            </div>
                          </Space>
                          <div
                            style={{
                              width: "100%",
                              height: "4px",
                              background: "#f0f0f0",
                              marginTop: "8px",
                              borderRadius: "2px",
                            }}
                          >
                            <div
                              style={{
                                width: `${(item.value / totalCount) * 100}%`,
                                height: "100%",
                                background:
                                  item.type === "交通顺畅" ||
                                  item.type === "sparse_traffic"
                                    ? "#52c41a"
                                    : item.type === "车辆拥堵" ||
                                      item.type === "交通拥堵" ||
                                      item.type === "dense_traffic" ||
                                      item.type === "congested_traffic"
                                    ? "#faad14"
                                    : item.type === "交通事故" ||
                                      item.type === "accident"
                                    ? "#f5222d"
                                    : item.type === "车辆起火" ||
                                      item.type === "fire"
                                    ? "#ff4d4f"
                                    : "#1890ff",
                                borderRadius: "2px",
                              }}
                            />
                          </div>
                        </Card>
                      </Col>
                    ))}
                  </Row>
                  <div
                    style={{
                      textAlign: "center",
                      marginTop: "16px",
                      color: "#888",
                      fontSize: "13px",
                    }}
                  >
                    <InfoCircleOutlined style={{ marginRight: "4px" }} />
                    基于历史预测记录的交通场景分析
                  </div>
                </div>
              )}
            </Card>
          </Col>

          <Col xs={24} md={12}>
            <Card
              title={
                <Space>
                  <LineChartOutlined />
                  <span>置信度趋势</span>
                </Space>
              }
              hoverable
              style={{ height: "100%" }}
              extra={
                <Select
                  value={recordCount}
                  onChange={setRecordCount}
                  style={{ width: 120 }}
                >
                  <Option value={20}>最近20条</Option>
                  <Option value={50}>最近50条</Option>
                  <Option value={100}>最近100条</Option>
                  <Option value={200}>最近200条</Option>
                </Select>
              }
            >
              {showLineChart ? (
                <>
                  <Line {...lineConfig} />
                  <div
                    style={{
                      textAlign: "center",
                      marginTop: "10px",
                      color: "#999",
                    }}
                  >
                    <span>
                      横轴表示历史记录序号，左侧为较早记录，右侧为最新记录
                    </span>
                  </div>
                </>
              ) : (
                <Empty
                  description="暂无数据"
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  style={{ margin: "50px 0" }}
                />
              )}
            </Card>
          </Col>
        </Row>
      </div>
    </div>
  );
};

export default TrafficAnalysis;
