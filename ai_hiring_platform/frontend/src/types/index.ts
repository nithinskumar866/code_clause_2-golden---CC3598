export interface ResumeMetadata {
  id: number;
  filename: string;
  upload_time: string;
  status: string;
}

export interface JobDescriptionMetadata {
  id: number;
  filename: string;
  upload_time: string;
  status: string;
}

export interface SystemStatusData {
  backend: string;
  database: string;
  llmProvider: string;
  vectorStore: string;
  embeddingModel: string;
  agentStatus: string;
}
