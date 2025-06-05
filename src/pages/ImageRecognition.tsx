import React, { useState } from "react";
import {
  Upload,
  Button,
  Card,
  Typography,
  Progress,
  Alert,
  message,
  Row,
  Col,
  Tag,
  Space,
  Divider,
  Tooltip,
  Spin,
  Statistic,
} from "antd";
import {
  InboxOutlined,
  DeleteOutlined,
  ExperimentOutlined,
  RocketOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  FireOutlined,
  CarOutlined,
  InfoCircleOutlined,
  PictureOutlined,
  LineChartOutlined,
} from "@ant-design/icons";
import type { RcFile } from "antd/es/upload";
import { predictImage } from "../api";
import type { PredictionResult } from "../types";

const { Dragger } = Upload;
const { Title, Text, Paragraph } = Typography;

// 交通实况类别详细信息
const trafficCategories = {
  sparse_traffic: {
    name: "交通顺畅",
    description: "车辆稀少，无交通拥堵，道路畅通无阻",
    color: "#52c41a",
    icon: <CheckCircleOutlined />,
    tag: "顺畅",
  },
  dense_traffic: {
    name: "交通拥堵",
    description: "车辆密集，行驶缓慢，但仍在移动",
    color: "#faad14",
    icon: <WarningOutlined />,
    tag: "拥堵",
  },
  accident: {
    name: "交通事故",
    description: "发生车辆碰撞或翻车等交通事故",
    color: "#f5222d",
    icon: <CarOutlined />,
    tag: "事故",
  },
  fire: {
    name: "车辆起火",
    description: "车辆着火，存在安全隐患",
    color: "#ff4d4f",
    icon: <FireOutlined />,
    tag: "起火",
  },
};

export const ImageRecognition: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [predictionResult, setPredictionResult] =
    useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const handleUpload = async (file: RcFile) => {
    const isImage = file.type.startsWith("image/");
    if (!isImage) {
      message.error("请上传图片文件");
      return false;
    }

    const isLessThan10M = file.size / 1024 / 1024 < 10;
    if (!isLessThan10M) {
      message.error("图片必须小于10MB");
      return false;
    }

    setFile(file);
    setImageUrl(URL.createObjectURL(file));
    setError(null);
    setPredictionResult(null);
    setProgress(0);
    return false;
  };

  const analyzeImage = async () => {
    if (!file) {
      message.error("请先上传图片");
      return;
    }

    setLoading(true);
    setAnalyzing(true);
    setError(null);

    // 模拟分析进度
    const progressInterval = setInterval(() => {
      setProgress((prevProgress) => {
        if (prevProgress >= 90) {
          clearInterval(progressInterval);
          return prevProgress;
        }
        return prevProgress + Math.floor(Math.random() * 10);
      });
    }, 300);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const result = await predictImage(formData);
      setPredictionResult(result);
      setProgress(100);
    } catch (error) {
      console.error("预测错误:", error);
      if (error && typeof error === "object" && "response" in error) {
        const response = error.response;
        if (response && typeof response === "object" && "data" in response) {
          const data = response.data;
          if (data && typeof data === "object") {
            if (
              "detail" in data &&
              typeof data.detail === "object" &&
              data.detail &&
              "message" in data.detail
            ) {
              const errorMessage = data.detail.message;
              if (typeof errorMessage === "string") {
                setError(errorMessage);
              } else {
                setError("交通实况分析失败，请重试");
              }
            } else if ("detail" in data && typeof data.detail === "string") {
              setError(data.detail);
            } else {
              setError("交通实况分析失败，请重试");
            }
          } else {
            setError("交通实况分析失败，请重试");
          }
        } else {
          setError("交通实况分析失败，请重试");
        }
      } else {
        setError("交通实况分析失败，请重试");
      }
    } finally {
      clearInterval(progressInterval);
      setLoading(false);
      setTimeout(() => {
        setAnalyzing(false);
      }, 500);
    }
  };

  const onPaste = (event: React.ClipboardEvent) => {
    const items = event.clipboardData?.items;
    if (items) {
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") !== -1) {
          const file = items[i].getAsFile();
          if (file) {
            handleUpload(file as RcFile);
          }
        }
      }
    }
  };

  const clearImage = () => {
    setImageUrl(null);
    setFile(null);
    setPredictionResult(null);
    setError(null);
    setProgress(0);
  };

  // 交通状况类别映射
  const translatePrediction = (prediction: string): string => {
    return (
      trafficCategories[prediction as keyof typeof trafficCategories]?.name ||
      prediction
    );
  };

  // 获取预测类别的详细信息
  const getPredictionCategory = (prediction: string) => {
    return (
      trafficCategories[prediction as keyof typeof trafficCategories] || {
        name: prediction,
        description: "未知交通状况",
        color: "#1890ff",
        icon: <InfoCircleOutlined />,
        tag: "未知",
      }
    );
  };

  // 获取模型分析阶段
  const getAnalysisStage = () => {
    if (progress < 30) return "图像预处理";
    if (progress < 60) return "特征提取";
    if (progress < 90) return "神经网络分析";
    return "生成结果";
  };

  return (
    <div style={{ width: "100%", padding: "20px" }}>
      <Card
        title={
          <Space>
            <RocketOutlined style={{ fontSize: "18px" }} />
            <span>交通实况智能识别</span>
            <Tooltip title="基于EfficientNet-B4深度学习模型">
              <Tag color="#108ee9">EfficientNet</Tag>
            </Tooltip>
          </Space>
        }
        style={{ width: "100%", borderRadius: "8px" }}
        bodyStyle={{ padding: "24px" }}
        extra={
          <Space>
            <Tooltip title="支持JPEG, PNG格式">
              <Tag icon={<PictureOutlined />} color="default">
                图像识别
              </Tag>
            </Tooltip>
            <Tooltip title="基于深度学习的交通场景分析">
              <Tag icon={<ExperimentOutlined />} color="blue">
                AI驱动
              </Tag>
            </Tooltip>
          </Space>
        }
      >
        <Row gutter={[24, 24]}>
          <Col xs={24} lg={12}>
            <div
              style={{
                width: "100%",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
              onPaste={onPaste}
              tabIndex={0}
            >
              {/* 上传区域 */}
              {!imageUrl && (
                <Dragger
                  showUploadList={false}
                  beforeUpload={handleUpload}
                  style={{ width: "100%", height: "280px" }}
                >
                  <p className="ant-upload-drag-icon">
                    <InboxOutlined
                      style={{ fontSize: "48px", color: "#1890ff" }}
                    />
                  </p>
                  <p className="ant-upload-text" style={{ fontSize: "18px" }}>
                    请上传交通图片进行智能识别
                  </p>
                  <p className="ant-upload-hint">
                    支持拖拽上传或点击选择文件（也可以直接粘贴图片）
                  </p>
                  <Divider style={{ margin: "12px 0" }}>
                    <Text type="secondary">
                      EfficientNet 深度学习模型提供支持
                    </Text>
                  </Divider>
                </Dragger>
              )}

              {/* 图片预览 */}
              {imageUrl && (
                <div
                  style={{
                    marginTop: "20px",
                    textAlign: "center",
                    width: "100%",
                  }}
                >
                  <div
                    style={{ position: "relative", display: "inline-block" }}
                  >
                    <img
                      src={imageUrl}
                      alt="交通实况预览"
                      style={{
                        maxWidth: "100%",
                        maxHeight: "300px",
                        borderRadius: "8px",
                        boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
                      }}
                    />
                    <Button
                      icon={<DeleteOutlined />}
                      onClick={clearImage}
                      style={{
                        position: "absolute",
                        top: "8px",
                        right: "8px",
                        background: "rgba(255, 255, 255, 0.8)",
                        borderRadius: "50%",
                        width: "36px",
                        height: "36px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        border: "none",
                        boxShadow: "0 2px 6px rgba(0, 0, 0, 0.15)",
                      }}
                    />
                  </div>
                  <div style={{ marginTop: "16px" }}>
                    <Button
                      type="primary"
                      onClick={analyzeImage}
                      loading={loading}
                      size="large"
                      icon={<ExperimentOutlined />}
                      style={{
                        borderRadius: "6px",
                        height: "44px",
                        fontSize: "16px",
                      }}
                    >
                      {loading ? "正在分析交通实况..." : "分析交通实况"}
                    </Button>
                  </div>
                </div>
              )}

              {/* 错误信息 */}
              {error && (
                <Alert
                  message={error}
                  type="error"
                  style={{ marginTop: "20px", width: "100%" }}
                />
              )}
            </div>
          </Col>

          <Col xs={24} lg={12}>
            {analyzing && !predictionResult ? (
              <div style={{ textAlign: "center", padding: "20px 0" }}>
                <Card
                  style={{
                    borderRadius: "8px",
                    marginBottom: "20px",
                    background:
                      "linear-gradient(135deg, #f5f7fa 0%, #e4efe9 100%)",
                  }}
                >
                  <Spin spinning={true} size="large" />
                  <div style={{ marginTop: "16px" }}>
                    <Progress
                      percent={progress}
                      status="active"
                      strokeColor={{
                        "0%": "#108ee9",
                        "100%": "#87d068",
                      }}
                    />
                    <div style={{ marginTop: "16px" }}>
                      <Tag
                        color="blue"
                        icon={<RocketOutlined />}
                        style={{ padding: "4px 8px" }}
                      >
                        {getAnalysisStage()}
                      </Tag>
                    </div>
                    <Paragraph style={{ marginTop: "16px" }}>
                      EfficientNet模型正在处理您的图像，通过深度神经网络提取特征并进行分类...
                    </Paragraph>
                  </div>
                </Card>

                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Card size="small">
                      <Statistic
                        title="识别精度"
                        value="93.5%"
                        valueStyle={{ color: "#3f8600" }}
                        prefix={<CheckCircleOutlined />}
                      />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card size="small">
                      <Statistic
                        title="模型版本"
                        value="B1"
                        valueStyle={{ color: "#1890ff" }}
                        prefix={<ExperimentOutlined />}
                      />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card size="small">
                      <Statistic
                        title="类别数量"
                        value={4}
                        valueStyle={{ color: "#722ed1" }}
                        prefix={<PictureOutlined />}
                      />
                    </Card>
                  </Col>
                </Row>
              </div>
            ) : predictionResult && !loading ? (
              <div>
                <Card
                  title={
                    <Space>
                      <span>识别结果</span>
                      <Tag color="blue">EfficientNet B1</Tag>
                    </Space>
                  }
                  style={{ borderRadius: "8px", marginBottom: "20px" }}
                >
                  <div style={{ textAlign: "center" }}>
                    <Tag
                      color={
                        getPredictionCategory(predictionResult.prediction).color
                      }
                      icon={
                        getPredictionCategory(predictionResult.prediction).icon
                      }
                      style={{
                        padding: "8px 16px",
                        fontSize: "18px",
                        marginBottom: "16px",
                      }}
                    >
                      {translatePrediction(predictionResult.prediction)}
                    </Tag>

                    <div style={{ width: "100%", margin: "0 auto 16px" }}>
                      <Progress
                        type="dashboard"
                        percent={Math.round(predictionResult.confidence * 100)}
                        status="active"
                        strokeColor={{
                          "0%": "#f5222d", // 红色
                          "40%": "#faad14", // 黄色
                          "80%": "#52c41a", // 绿色
                        }}
                        format={(percent) => (
                          <span style={{ fontSize: "18px" }}>
                            {percent?.toFixed(1)}%
                            <div style={{ fontSize: "12px", color: "#888" }}>
                              置信度
                            </div>
                          </span>
                        )}
                      />
                    </div>
                  </div>
                </Card>

                <Card
                  title={
                    <span>
                      <InfoCircleOutlined /> 模型信息
                    </span>
                  }
                  style={{ borderRadius: "8px" }}
                  size="small"
                >
                  <Paragraph>
                    <Text strong>EfficientNet</Text>{" "}
                    是一种先进的卷积神经网络架构，由Google设计，
                    它通过优化网络宽度、深度和分辨率的平衡，实现了更高的精度和效率。
                    B1版本具有780万参数，比传统模型更轻量但精度更高。
                  </Paragraph>

                  <Divider style={{ margin: "12px 0" }} />

                  <Row gutter={[16, 16]}>
                    <Col span={12}>
                      <Statistic
                        title="参数量"
                        value="780万"
                        valueStyle={{ fontSize: "14px" }}
                        prefix={<LineChartOutlined />}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="深度"
                        value="237层"
                        valueStyle={{ fontSize: "14px" }}
                        prefix={<ExperimentOutlined />}
                      />
                    </Col>
                  </Row>
                </Card>
              </div>
            ) : (
              <div
                style={{
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Card
                  style={{
                    width: "100%",
                    borderRadius: "8px",
                    background:
                      "linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%)",
                    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.05)",
                  }}
                >
                  <div style={{ textAlign: "center", padding: "20px" }}>
                    <ExperimentOutlined
                      style={{
                        fontSize: "48px",
                        color: "#1890ff",
                        marginBottom: "16px",
                      }}
                    />
                    <Title level={4}>EfficientNet 深度学习模型</Title>
                    <Paragraph>
                      本系统采用先进的EfficientNet
                      B1模型识别交通场景。该模型通过优化设计，
                      在保持轻量化的同时提供了优异的识别性能，可识别交通顺畅、拥堵、事故和车辆起火等多种场景。
                    </Paragraph>

                    <Divider>主要特点</Divider>

                    <Row gutter={[16, 16]}>
                      {Object.entries(trafficCategories).map(
                        ([key, category]) => (
                          <Col key={key} span={12}>
                            <Card size="small" style={{ textAlign: "left" }}>
                              <Space>
                                {category.icon}
                                <Text strong>{category.name}</Text>
                                <Tag color={category.color}>{category.tag}</Tag>
                              </Space>
                              <Paragraph
                                style={{ marginTop: "8px", fontSize: "12px" }}
                              >
                                {category.description}
                              </Paragraph>
                            </Card>
                          </Col>
                        )
                      )}
                    </Row>
                  </div>
                </Card>
              </div>
            )}
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default ImageRecognition;
