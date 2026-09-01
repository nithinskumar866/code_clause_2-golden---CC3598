using JobPortal.Api.Options;
using JobPortal.Api.Services.Rag.Data;
using JobPortal.Api.Services.Rag.Ingestion;
using JobPortal.Api.Services.Rag.Vectors;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Qdrant.Client;

namespace JobPortal.Api.Services.Rag;

/// <summary>
/// The RAG mode's entire wiring, in one call.
///
/// Everything the feature needs is registered here rather than scattered through
/// Program.cs, so the whole thing is added by one line and removed by deleting
/// it. That is the contract of building this behind a third button: the existing
/// modes must not acquire a dependency on any of it.
/// </summary>
public static class RagServiceCollectionExtensions
{
    public static IServiceCollection AddRagMode(
        this IServiceCollection services, IConfiguration configuration)
    {
        services.Configure<RagOptions>(configuration.GetSection(RagOptions.SectionName));

        var options = configuration.GetSection(RagOptions.SectionName).Get<RagOptions>() ?? new RagOptions();

        // Disabled, or not configured: register nothing. A half-registered feature
        // whose services resolve but whose stores are absent fails at the first
        // request instead of at startup, which is the worse of the two.
        if (!options.Enabled || string.IsNullOrWhiteSpace(options.PostgresConnectionString))
        {
            return services;
        }

        services.AddDbContext<RagDbContext>(db => db.UseNpgsql(
            options.PostgresConnectionString,
            npgsql => npgsql
                // A transient network blip during a 1.5M-chunk ingestion run must
                // not park a document as failed.
                .EnableRetryOnFailure(maxRetryCount: 3, maxRetryDelay: TimeSpan.FromSeconds(5), null)
                .CommandTimeout(60)));

        // One client for the process. It holds a gRPC channel, and building one
        // per request would open a connection per request.
        services.AddSingleton(_ => new QdrantClient(
            host: options.QdrantHost,
            port: options.QdrantPort,
            https: options.QdrantUseTls,
            apiKey: string.IsNullOrWhiteSpace(options.QdrantApiKey) ? null : options.QdrantApiKey));

        services.AddSingleton<IEmbeddingDimensionSource, OllamaEmbeddingDimension>();
        services.AddSingleton<QdrantBootstrap>();
        services.AddScoped<IRagVectorStore, QdrantRagVectorStore>();

        // Chunking is pure and holds no state; the queue must be shared by every
        // producer and the single consumer.
        services.AddSingleton<DocumentChunker>();
        services.AddSingleton<RagIngestionQueue>();

        services.AddScoped<Retrieval.IRagRetriever, Retrieval.RagRetriever>();
        services.AddScoped<Agent.RagEvaluationAgent>();

        // The agent reuses the scoring arithmetic through JobEvaluationService's
        // Compose seam, which is not on IJobEvaluationService — that interface
        // describes "evaluate a pair with a model", and Compose calls no model.
        // Program.cs registers only the interface, so the concrete type is added
        // here rather than changing a registration the other modes depend on.
        services.AddScoped<Matching.JobEvaluationService>();
        services.AddScoped<IRagMatchingService, RagMatchingService>();
        services.AddScoped<RagIngestionService>();
        services.AddHostedService<RagIngestionWorker>();

        return services;
    }
}

/// <summary>
/// The embedding width, read from the same options the embedder itself uses.
///
/// Deliberately not a separate Rag setting: two places declaring the vector width
/// is two places to disagree, and the disagreement only surfaces as every search
/// scoring zero.
/// </summary>
public class OllamaEmbeddingDimension : IEmbeddingDimensionSource
{
    private readonly OllamaOptions _ollama;

    public OllamaEmbeddingDimension(IOptions<OllamaOptions> ollama) => _ollama = ollama.Value;

    public int Dimension => _ollama.EmbedDimension;
}
