// 用户信息类型
export interface User {
  username: string;
  full_name?: string;
  role: string;
  is_active: boolean;
}

// 登录表单类型
export interface LoginForm {
  username: string;
  password: string;
}

// 预测结果类型
export interface PredictionResult {
  prediction: string;
  confidence: number;
  image_url?: string;
  id?: string;
}

// 历史记录类型
export interface HistoryRecord {
  id: string;
  image_url: string;
  prediction: string;
  confidence: number;
  timestamp: string;
  username?: string;
}

// 预测历史记录项（API响应格式）
export interface PredictionHistoryItem {
  _id: string;
  image_url: string;
  prediction: string;
  confidence: number;
  timestamp: string;
  username?: string;
  user_id?: string;
  image_path?: string;
  model_name?: string;
  created_at?: string;
}

// 用户更新类型
export interface UserUpdate {
  full_name?: string;
  password?: string;
  is_active?: boolean;
  role?: string;
}

export interface TrafficStatistics {
  prediction_types: Record<string, number>;
  monthly_counts: Record<string, number>;
  confidence_ranges: Record<string, number>;
  avg_confidence: number;
  total_count: number;
}

export interface ConfidenceTrendItem {
  date?: string;
  avg_confidence: number;
  id?: string;
  created_at?: string;
  index?: number;
}
