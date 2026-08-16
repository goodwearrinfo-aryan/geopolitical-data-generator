import React, { useEffect } from "react"
import { useStore } from "./stores/useStore"
import { useCelery } from "./hooks/useCelery"

export default function App() {
  const {
    // Original store
    scenarios,
    jobs,
    selectedScenario,
    loadInitialData,
    runScenario,
    cancelScenario,
    exportJob,
    
    // Celery integration
    tasks,
    fetchTask,
    setTask,
  } = useStore()

  // Load initial data
  useEffect(() => {
    loadInitialData()
    // Fetch Celery task statuses on mount
    Object.values(tasks).forEach((task) => {
      if (task.status === "STARTED" || task.status === "PENDING") {
        fetchTask(task.id)
      }
    })
  }, [])

  // Periodic Celery task polling
  useEffect(() => {
    // Poll all active tasks every 3 seconds
    const interval = setInterval(() => {
      Object.keys(tasks).forEach((id) => {
        const task = tasks[id]
        if (task.status in ["PENDING", "STARTED", "PROGRESS"]) {
          fetchTask(id)
        }
      })
    }, 3000)
    return () => clearInterval(interval)
  }, [tasks])

  return (
    <div style={{ padding: "1rem" }} fontFamily="system-ui, sans-serif">
      <div style={{ fontWeight: 600, fontSize: "1.25rem", marginBottom: "1rem" }}>
        Geopolitical Data Generator v1.0.0
      </div>

      {/* Scenarios Section */}
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "0.5rem", padding: "1rem", marginBottom: "1rem" }}>
        <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Scenarios</div>
        <div>
          {scenarios.length === 0
            ? <div>No scenarios available</div>
            : scenarios.map((s) => (
                <div key={s.id} style={{ marginBottom: "0.25rem" }}>
                  <span style={{ marginRight: "0.5rem" }}>{s.name}</span>
                  <button
                    onClick={() => runScenario(s.id)}
                    style={{ 
                      marginLeft: "0.5rem", padding: "0.25rem 0.5rem", fontSize: "0.75rem"
                    }}
                  >
                    Run (Celery)
                  </button>
                </div>
              ))}
        </div>
      </div>

      {/* Jobs / Celery Tasks Section */}
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "0.5rem", padding: "1rem", marginBottom: "1rem" }}>
        <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Celery Tasks</div>
        <div>
          {Object.keys(tasks).length === 0
            ? <div>No active Celery tasks</div>
            : Object.entries(tasks).map(([id, task]) => (
                <div
                  key={id}
                  style={{
                    marginBottom: "0.75rem",
                    paddingBottom: "0.5rem",
                    borderBottom: "1px solid #e2e8f0",
                  }}
                >
                  <span style={{ marginRight: "0.5rem" }}>{id.substring(0, 8)}...</span>
                  <div style={{ 
                    width: "150px", height: "8px", 
                    backgroundColor: "#e2e8f0", borderRadius: "4px", overflow: "hidden",
                    marginBottom: "0.25rem"
                  }}>
                    <div
                      style={{
                        width: `${task.progress}%`, height: "100%", 
                        backgroundColor: task.progress > 50 ? "#10b981" : task.progress > 25 ? "#f59e0b" : "#ef4444",
                        transition: "width 0.3s ease", height: "100%"
                      }}
                    ></div>
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "#64748b" }}>
                    {task.status} — {task.progress}%
                  </div>
                  {task.error && (
                    <div style={{ marginTop: "0.25rem", fontSize: "0.65rem", color: "#ef4444", backgroundColor: "#fef2f2", padding: "0.25rem", borderRadius: "4px" }}>
                      {task.error}
                    </div>
                  )}
                  <button
                    onClick={() => fetchTask(id)}
                    style={{ marginLeft: "0.5rem", padding: "0.25rem 0.4rem", fontSize: "0.6rem", backgroundColor: "#3b82f6", color: "white" }}
                    title="Refresh status">
                    🔄
                  </button>
                </div>
              ))}
        </div>
      </div>

      {/* Exports Section */}
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "0.5rem", padding: "1rem", marginBottom: "1rem" }}>
        <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Exports</div>
        <div>
          {Object.keys(tasks).length > 0 && Object.values(tasks).some(t => t.status === "SUCCESS" && t.result)}
            ? <div>Exports complete - check dashboard</div>
            : <div>No recent exports</div>
        </div>
      </div>

      {/* Controls */}
      <div style={{ marginTop: "1rem", fontSize: "0.75rem", color: "#64748b" }}>
        <span>Refresh <button onClick={() => window.location.reload()">auto</button></span>
      </div>
    </div>
  )
}
