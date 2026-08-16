import { create } from "zustand"
import { persist, createJSONStorage } from "zustand/middleware/persist"

export type CeleryTaskStatus = 
  | "PENDING"
  | "STARTED" 
  | "PROGRESS"
  | "SUCCESS"
  | "FAILURE"
  | "RETRY"

export type CeleryTask = {
  id: string
  status: CeleryTaskStatus
  progress: number
  result?: any
  error?: string
  description?: string
}

type CeleryStore = {
  tasks: Record<string, CeleryTask>
  setTask: (id: string, task: CeleryTask) => void
  fetchTask: (id: string) => void
  cancelTask: (id: string) => void
}

export const useCelery = create<Store & CeleryStore>(
  persist(
    (set, get) => ({
      tasks: {},
      setTask: (id, task) => set({ tasks: { ...get().tasks, [id]: task } }),
      fetchTask: async (id: string) => {
        try {
          const resp = await fetch(`/api/v1/jobs/${id}`)
          const data = await resp.json()
          
          // Map API status to Celery status
          let celeryStatus: CeleryTaskStatus = "PENDING"
          if (data.status === "completed" || data.status === "completed") celeryStatus = "SUCCESS"
          else if (data.status === "failed" || data.status === "error") celeryStatus = "FAILURE"
          else if (data.status === "running" || data.status === "pending") celeryStatus = "STARTED"
          else if (data.status === "progress") celeryStatus = "PROGRESS"
          
          set({ tasks: { ...get().tasks, [id]: { id, status: celeryStatus, progress: data.progress || 0, result: data.result, error: data.error } } })
        } catch (e) {
          set({ tasks: { ...get().tasks, [id]: { id, status: "FAILURE", error: (e as Error).message } } })
        }
      },
      cancelTask: (id: string) => {
        set({ tasks: Object.fromEntries(
          Object.entries(get().tasks).filter(([key]) => key !== id)
        )})
      },
    }),
    { name: "geopolitical-storage" },
    { skipHydration: true }
  )
