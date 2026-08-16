import axios from "axios"

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api/v1"

export const api = axios.create({
  baseURL: API_BASE,
  headers: {
    "Content-Type": "application/json",
  },
})

export const fetchScenarios = async () => {
  const { data } = await api.get("/scenarios/")
  return data
}

export const fetchJobs = async () => {
  const { data } = await api.get("/jobs/")
  return data
}

export const runScenario = async (scenarioId: string) => {
  const { data } = await api.post("/api/v1/scenarios/", { scenario_id: scenarioId })
  return data
}

export const getJobStatus = async (jobId: string) => {
  const { data } = await api.get(`/jobs/${jobId}`)
  return data
}

export const exportResults = async (jobId: string, format: "parquet" | "csv" | "geojson") => {
  const { data, status } = await api.get(`/api/v1/exports/${jobId}?format=${format}`, {
    responseType: "blob",
  })
  return { data, status }
}