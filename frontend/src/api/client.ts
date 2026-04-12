import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 300_000, // 5 min — upscaling can be slow
});
