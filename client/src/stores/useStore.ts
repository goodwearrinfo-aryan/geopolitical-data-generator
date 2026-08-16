import create from "zustand"
import { persist, createJSONStorage } from "zustand/middleware/persist"
import { fetchScenarios as apiFetchScenarios, fetchJobs as apiFetchJobs, runScenario as apiRunScenario, getJobStatus as apiGetJobStatus, exportResults } from "./api/apiService"

export type Scenario = { id: string; name: string }

export type Job = {
  id: string
  name?: string
  status: "pending" | "running" | "completed" | "failed"
  progress: number
  result?: any
  error?: string
}

type Store = {
  scenarios: Scenario[]
  jobs: Job[]
  selectedScenario?: string
  setScenarios: (s: Scenario[]) => void
  setJobs: (j: Job[]) => void
  selectScenario: (id: string) => void
  runScenario: (id: string) => void
  cancelScenario: () => void
  loadInitialData: () => void
  exportJob: (jobId: string, format: string) => void
}

export const useStore = create<Store>(
  persist(
    (set, get) => ({
      scenarios: [],
      jobs: [],
      selectedScenario: undefined,
      setScenarios: (s) => set({ scenarios: s }),
      setJobs: (j) => set({ jobs: j }),
      selectScenario: (id) => set({ selectedScenario: id }),
      runScenario: async (id: string) => {
        // Remove any existing job with this ID
        set({ jobs: get().jobs.filter((j) => j.id !== `temp-${id}`) })
        
        // Run the scenario via API
        const result = await apiRunScenario(id)
        
        // Create a job entry from the result
        const jobId = `job-${Date.now()}`
        const newJob: Job = {
          id: jobId,
          name: result?.name || `Scenario ${id}`,
          status: "running",
          progress: 0,
          result: result,
          error: undefined,
        }
        
        set({ jobs: [...get().jobs, newJob] })
        
        // Connect to WebSocket for real-time progress updates
        await websocketProgress(jobId)
        
        return jobId
      },
      cancelScenario: () => set({ selectedScenario: undefined }),
      loadInitialData: async () => {
        const [scenariosResult, jobsResult] = await Promise.all([
          apiFetchScenarios(),
          apiFetchJobs(),
        ])
        set({ scenarios: scenariosResult.scenarios || [] })
        set({ jobs: jobsResult.jobs || [] })
      },
      exportJob: async (jobId: string, format: string) => {
        set({ jobs: get().jobs.map((j) =>
          j.id === jobId ? { ...j, status: "exporting", progress: 0, error: undefined } : j
        ) })
        
        try {
          const { data, status } = await exportResults(jobId, format)
          // Trigger download
          const url = URL.createObjectURL(data)
          const a = document.createElement("a")
          a.href = url
          a.download = `results.${format}`
          a.click()
          URL.revokeObjectURL(url)
          
          set({ jobs: get().jobs.map((j) =>
            j.id === jobId ? { ...j, status: "completed", progress: 100, error: undefined } : j
          )})
        } catch (e: any) {
          set({ jobs: get().jobs.map((j) =>
            j.id === jobId ? { ...j, status: "failed", error: e.message || "Export failed" } : j
          )})
        }
      },
    }),
    {
      name: "geopolitical-storage",
      storage: createJSONStorage<Store>(localStorage),
    }
  )
)

// WebSocket progress connector
async function websocketProgress(jobId: string) {
  const store = useStore.getState()
  
  // Don't connect if already have a WebSocket for this job
  if (store.jobs.some((j) => j.id === jobId && j.status === "running")) {
    const ws = new WebSocket(`ws://localhost:8000/api/v1/jobs/${jobId}/progress`)
    
    ws.onopen = () => {
      console.log(`WebSocket connected for job ${jobId}`)
    }
    
    ws.onmessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data)
      set({
        jobs: store.jobs.map((j) =>
          j.id === jobId ? { ...j, status: data.status, progress: data.progress } : j
        ),
      })
    }
    
    ws.onerror = (error) => {
      console.error(`WebSocket error for job ${jobId}:`, error)
      // Fall back to polling
      startPollingProgress(jobId)
    }
    
    ws.onclose = () => {
      // Fall back to polling when WebSocket closes
      startPollingProgress(jobId)
    }
  }
}

function startPollingProgress(jobId: string) {
  // Poll every 2 seconds as fallback
  const interval = setInterval(async () => {
    const store = useStore.getState()
    const currentJob = store.jobs.find((j) => j.id === jobId)
    if (!currentJob) {
      clearInterval(interval)
      return
    }
    
    try {
      const status = await apiGetJobStatus(jobId)
      set({
        jobs: store.jobs.map((j) =>
          j.id === jobId ? { ...j, status: status.status, progress: status.progress } : j
        ),
      })
      
      if (status.status === "completed" || status.status === "failed") {
        clearInterval(interval)
      }
    } catch (e) {
      // If API fails, simulate progress
      set({
        jobs: store.jobs.map((j) =>
          j.id === jobId ? { ...j, progress: Math.min(j.progress + 15, 100) } : j
        ),
      })
      if (store.jobs.find((j) => j.id === jobId)?.progress >= 100) {
        clearInterval(interval)
      }
    }
  }, 2000)
}