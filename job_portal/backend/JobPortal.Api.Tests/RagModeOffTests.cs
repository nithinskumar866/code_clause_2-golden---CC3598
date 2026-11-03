using JobPortal.Api.Controllers;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Rag;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace JobPortal.Api.Tests;

/// <summary>
/// The portal must run with the RAG mode switched off. The chat orchestrator depends on
/// the matching service, so registering nothing stopped the whole API from starting on
/// any deployment without Postgres.
/// </summary>
public class RagModeOffTests
{
    private static IServiceProvider ServicesWithNoRagSettings() =>
        new ServiceCollection()
            .AddRagMode(new ConfigurationBuilder().Build())
            .BuildServiceProvider();

    [Fact]
    public async Task Matching_service_resolves_and_reports_itself_unavailable()
    {
        using var scope = ServicesWithNoRagSettings().CreateScope();

        var rag = scope.ServiceProvider.GetRequiredService<IRagMatchingService>();

        Assert.False(await rag.IsAvailableAsync());
    }

    [Fact]
    public async Task Status_says_unavailable_rather_than_failing()
    {
        var controller = new RagController(Microsoft.Extensions.Options.Options.Create(new RagOptions()));

        var response = await controller.Status(CancellationToken.None);

        var ok = Assert.IsType<OkObjectResult>(response.Result);
        var body = Assert.IsAssignableFrom<JobPortal.Api.Contracts.ApiResponse<RagController.RagStatus>>(ok.Value);
        Assert.False(body.Data!.Available);
    }

    [Fact]
    public async Task Reindex_refuses_with_503()
    {
        var controller = new RagController(Microsoft.Extensions.Options.Options.Create(new RagOptions()));

        var response = await controller.Reindex("job", 1, CancellationToken.None);

        var result = Assert.IsAssignableFrom<ObjectResult>(response.Result);
        Assert.Equal(503, result.StatusCode);
    }
}

/// <summary>
/// <c>SkillTransferService</c> reads and writes this table on every reasoned evaluation,
/// and the schema initializer never created it.
/// </summary>
public class SkillVerdictCacheSchemaTests
{
    [Fact]
    public async Task The_verdict_cache_can_be_written_and_read_on_a_new_database()
    {
        var path = Path.Combine(Path.GetTempPath(), $"portal-schema-{Guid.NewGuid():N}.db");
        var options = new DbContextOptionsBuilder<PortalDbContext>()
            .UseSqlite($"Data Source={path};Pooling=False")
            .Options;

        try
        {
            await using var db = new PortalDbContext(options);
            await PortalSchemaInitializer.InitializeAsync(db, NullLogger.Instance);

            db.Jobs.Add(new PortalJob { Id = 1, Title = "DevOps Engineer" });
            db.SkillVerdictCache.Add(new PortalSkillVerdictCache
            {
                JobId = 1,
                CandidateSkillSetHash = "hash",
                RequiredSkill = "Kubernetes",
                Verdict = "Transferable",
                EvidenceSkill = "Docker",
                Confidence = 0.7,
            });
            await db.SaveChangesAsync();

            var stored = await db.SkillVerdictCache.SingleAsync();
            Assert.Equal("Docker", stored.EvidenceSkill);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
