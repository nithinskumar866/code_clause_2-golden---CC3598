using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Jobs;
using JobPortal.Api.Services.Llm;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace JobPortal.Api.Controllers;

[ApiController]
[Route("api/health")]
public class HealthController : ControllerBase
{
    private readonly PortalDbContext _db;
    private readonly IOllamaClient _llm;
    private readonly IJobService _jobs;

    public HealthController(PortalDbContext db, IOllamaClient llm, IJobService jobs)
    {
        _db = db;
        _llm = llm;
        _jobs = jobs;
    }

    /// <summary>
    /// What is actually working right now.
    ///
    /// Reports the degraded states explicitly rather than returning a bare "ok":
    /// the portal runs with no LLM and with no embedding model, but it behaves
    /// differently in each case, and "why did this candidate not surface" is
    /// unanswerable without knowing which mode produced the result.
    /// </summary>
    [HttpGet]
    public async Task<ActionResult<ApiResponse<HealthDto>>> Get(CancellationToken ct)
    {
        var status = await _jobs.GetIndexStatusAsync(ct);
        var pythonVisible = await PortalSchemaInitializer.TableExistsAsync(_db, "job_descriptions", ct);

        // Only probed when a base URL is configured, so a health check on a
        // deliberately offline install is instant rather than a timeout.
        var models = _llm.ChatAvailable
            ? await _llm.ListModelsAsync(ct)
            : Array.Empty<string>();

        var health = new HealthDto(
            Status: "ok",
            Database: _db.Database.GetDbConnection().DataSource ?? "unknown",
            PythonTablesVisible: pythonVisible,
            ChatModelConfigured: _llm.ChatAvailable,
            ChatModel: _llm.ChatModel,
            EmbeddingModelConfigured: _llm.EmbeddingsAvailable,
            EmbeddingModel: status.ActiveModel,
            SemanticMatching: status.SemanticMatching,
            AvailableModels: models,
            JobCount: status.TotalJobs,
            IndexedJobCount: status.IndexedJobs);

        return Ok(ApiResponse<HealthDto>.Ok(health));
    }
}
