using JobPortal.Api.Data;
using JobPortal.Api.Hubs;
using JobPortal.Api.Options;
using JobPortal.Api.Services.Applications;
using JobPortal.Api.Services.Chat;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Jobs;
using JobPortal.Api.Services.Llm;
using JobPortal.Api.Services.Matching;
using JobPortal.Api.Services.Rag;
using JobPortal.Api.Services.Rag.Data;
using JobPortal.Api.Services.Rag.Vectors;
using JobPortal.Api.Services.Resumes;
using JobPortal.Api.Services.Vectors;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);

// Secrets live in a .env file next to the project rather than in appsettings.json,
// so an API key cannot be committed by editing config. Environment variables still
// win, which is what a container deployment needs.
DotEnv.Load(Path.Combine(builder.Environment.ContentRootPath, ".env"), builder.Configuration);

builder.Services.Configure<PortalOptions>(builder.Configuration.GetSection(PortalOptions.SectionName));
builder.Services.Configure<OllamaOptions>(builder.Configuration.GetSection(OllamaOptions.SectionName));
builder.Services.Configure<MatchingOptions>(builder.Configuration.GetSection(MatchingOptions.SectionName));
builder.Services.Configure<ScreeningOptions>(builder.Configuration.GetSection(ScreeningOptions.SectionName));

// The RAG scoring mode: Postgres for chunks, Qdrant for vectors. Registers
// nothing when disabled or unconfigured, so the two existing modes are unaffected
// on a deployment without those stores.
builder.Services.AddRagMode(builder.Configuration);

var portal = builder.Configuration.GetSection(PortalOptions.SectionName).Get<PortalOptions>() ?? new PortalOptions();
PortalPaths.Resolve(portal, builder.Environment.ContentRootPath);
builder.Services.PostConfigure<PortalOptions>(o => PortalPaths.Resolve(o, builder.Environment.ContentRootPath));

var ollama = builder.Configuration.GetSection(OllamaOptions.SectionName).Get<OllamaOptions>() ?? new OllamaOptions();

// -- data -------------------------------------------------------------------

builder.Services.AddDbContext<PortalDbContext>(options =>
    options.UseSqlite($"Data Source={portal.DatabasePath}"));

// -- http clients -----------------------------------------------------------

builder.Services.AddHttpClient(OllamaClient.ChatHttpClient, client =>
    OllamaClient.Configure(client, ollama.BaseUrl, ollama.ApiKey, ollama.TimeoutSeconds));

builder.Services.AddHttpClient(OllamaClient.EmbedHttpClient, client =>
    OllamaClient.Configure(client, ollama.ResolvedEmbedBaseUrl, ollama.ResolvedEmbedApiKey, ollama.TimeoutSeconds));

// -- services ---------------------------------------------------------------

builder.Services.AddSingleton<IOllamaClient, OllamaClient>();

// Singletons because both hold process-wide state that exists precisely to avoid
// repeating expensive work across requests.
// Backs the evaluation cache: a reasoned judgement is expensive and deterministic,
// so the same résumé against the same posting is answered from here on a re-ask.
builder.Services.AddMemoryCache();

builder.Services.AddSingleton(new HashingEmbeddingService(512));
builder.Services.AddSingleton<JobVectorCache>();
builder.Services.AddSingleton<SkillVectorCache>();
builder.Services.AddSingleton<ChatQueryUnderstanding>();
// Singleton like the parser it wraps: it holds no per-request state, and the LLM
// client it calls is a singleton too.
builder.Services.AddSingleton<ChatTurnPlanner>();

builder.Services.AddScoped<IEmbeddingService, EmbeddingService>();
builder.Services.AddScoped<IVectorStore, SqliteVectorStore>();
builder.Services.AddScoped<IDocumentTextExtractor, DocumentTextExtractor>();
builder.Services.AddScoped<IJobExtractionService, JobExtractionService>();
builder.Services.AddScoped<IResumeProfileService, ResumeProfileService>();
builder.Services.AddScoped<IJobService, JobService>();
builder.Services.AddScoped<IJobSuggestionService, JobSuggestionService>();
builder.Services.AddScoped<IResumeService, ResumeService>();
builder.Services.AddScoped<ISkillSemanticsService, SkillSemanticsService>();
builder.Services.AddScoped<ISkillTransferService, SkillTransferService>();
 builder.Services.AddScoped<IResumeEvidenceService, ResumeEvidenceService>();
builder.Services.AddScoped<IJobEvaluationService, JobEvaluationService>();
builder.Services.AddScoped<IMatchingService, MatchingService>();
// The reasoned scorer sits ON TOP of the computed one — it uses it for retrieval
// and ordering, then replaces the verdict. Registered as a separate service rather
// than as a second IMatchingService so a caller always knows which it resolved.
builder.Services.AddScoped<IReasonedMatchingService, ReasonedMatchingService>();
// Pre-application screening. Self-contained in Services/Screening — nothing else
// depends on it, and removing this line disables the feature entirely.
builder.Services.AddScoped<JobPortal.Api.Services.Screening.IApplicationScreeningService,
                           JobPortal.Api.Services.Screening.ApplicationScreeningService>();
builder.Services.AddScoped<ICoverLetterService, CoverLetterService>();
builder.Services.AddScoped<IApplicationService, ApplicationService>();
// Scoped: reads the database through the request's DbContext.
builder.Services.AddScoped<ChatCapabilities>();
builder.Services.AddSingleton<AnswerComposer>();
builder.Services.AddScoped<IChatOrchestrator, ChatOrchestrator>();

// -- web --------------------------------------------------------------------

builder.Services.AddControllers();
builder.Services.AddSignalR();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

const string CorsPolicy = "portal-frontend";
builder.Services.AddCors(options => options.AddPolicy(CorsPolicy, policy =>
    policy.WithOrigins(portal.AllowedOrigins)
          .AllowAnyHeader()
          .AllowAnyMethod()
          // SignalR sends a credentialed negotiate request, which the browser
          // refuses against a wildcard origin. This is why AllowedOrigins is an
          // explicit list rather than "*".
          .AllowCredentials()));

var app = builder.Build();

// -- startup ----------------------------------------------------------------

using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<PortalDbContext>();
    var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();

    Directory.CreateDirectory(Path.GetDirectoryName(portal.DatabasePath)!);
    await PortalSchemaInitializer.InitializeAsync(db, logger);

    var options = scope.ServiceProvider.GetRequiredService<IOptions<OllamaOptions>>().Value;
    if (!options.Enabled)
    {
        logger.LogWarning(
            "No chat model configured (Ollama:BaseUrl / Ollama:ChatModel). The portal runs on its " +
            "deterministic path: extraction, matching and replies all work, but without written " +
            "reasoning.");
    }
    if (!options.EmbeddingsEnabled)
    {
        logger.LogWarning(
            "No embedding model configured (Ollama:EmbedModel). Matching falls back to the " +
            "deterministic embedder, which compares WORDING rather than MEANING. Semantic " +
            "matching needs an embedding endpoint.");
    }

    // -- RAG mode storage --
    //
    // Migrated and bootstrapped at startup, and failure here is NOT fatal: the
    // two existing scoring modes need neither store, and a Postgres that is slow
    // to accept connections must not stop the portal from serving.
    var rag = scope.ServiceProvider.GetRequiredService<IOptions<RagOptions>>().Value;
    if (rag.Enabled && !string.IsNullOrWhiteSpace(rag.PostgresConnectionString))
    {
        try
        {
            var ragDb = scope.ServiceProvider.GetRequiredService<RagDbContext>();
            await ragDb.Database.MigrateAsync();

            await scope.ServiceProvider.GetRequiredService<QdrantBootstrap>()
                .EnsureCollectionsAsync();

            logger.LogInformation("RAG mode ready: Postgres migrated, Qdrant collections present.");
        }
        catch (Exception ex)
        {
            logger.LogError(ex,
                "RAG storage is unavailable. The RAG scoring mode will report itself unavailable; " +
                "Fast score and AI evaluation are unaffected.");
        }
    }
    else
    {
        logger.LogInformation("RAG mode is disabled or unconfigured (Rag:Enabled / Rag:PostgresConnectionString).");
    }
}

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseCors(CorsPolicy);

// -- the built frontend, served from this same process ----------------------
//
// One origin for the API, the hub and the page. That removes CORS from the
// picture entirely, means the browser never needs a second base URL to guess at,
// and leaves a single port to expose when this is shared or tunnelled — the
// alternative is two public URLs that have to know about each other.
//
// Absent in development, where Vite serves the SPA and proxies nothing.
if (!string.IsNullOrWhiteSpace(portal.SpaPath) && Directory.Exists(portal.SpaPath))
{
    var spa = new PhysicalFileProvider(portal.SpaPath);

    app.UseDefaultFiles(new DefaultFilesOptions { FileProvider = spa });
    app.UseStaticFiles(new StaticFileOptions { FileProvider = spa });

    app.MapControllers();
    app.MapHub<ChatHub>(ChatHub.Route);

    // Anything not matched above is a client route. Served index.html so a deep
    // link or a refresh lands in the app rather than on a 404 — but only AFTER
    // the API and hub routes, so it can never swallow them.
    app.MapFallbackToFile("index.html", new StaticFileOptions { FileProvider = spa });

    app.Logger.LogInformation("Serving the frontend from {Path}", portal.SpaPath);
}
else
{
    app.MapControllers();
    app.MapHub<ChatHub>(ChatHub.Route);
}

app.Run();

/// <summary>
/// Turns configured paths into absolute ones and supplies defaults that point at
/// the Python platform this portal shares a database with.
/// </summary>
internal static class PortalPaths
{
    public static void Resolve(PortalOptions options, string contentRoot)
    {
        // job_portal/backend/JobPortal.Api -> repository root
        var repoRoot = Path.GetFullPath(Path.Combine(contentRoot, "..", "..", ".."));
        var pythonBackend = Path.Combine(repoRoot, "ai_hiring_platform", "backend");

        if (string.IsNullOrWhiteSpace(options.DatabasePath))
            options.DatabasePath = Path.Combine(pythonBackend, "hiring_platform.db");

        if (string.IsNullOrWhiteSpace(options.PythonStorageDir))
            options.PythonStorageDir = Path.Combine(pythonBackend, "storage");

        if (string.IsNullOrWhiteSpace(options.ResumeUploadDir))
            options.ResumeUploadDir = Path.Combine(contentRoot, "storage", "resumes");

        // Left blank by default: development serves the SPA from Vite, and pointing
        // at a stale dist/ would quietly serve yesterday's build alongside today's
        // API. Set it explicitly to host both from here.
        if (!string.IsNullOrWhiteSpace(options.SpaPath))
            options.SpaPath = Path.GetFullPath(options.SpaPath);

        options.DatabasePath = Path.GetFullPath(options.DatabasePath);
        options.PythonStorageDir = Path.GetFullPath(options.PythonStorageDir);
        options.ResumeUploadDir = Path.GetFullPath(options.ResumeUploadDir);
    }
}

/// <summary>
/// Minimal .env reader.
///
/// Keys use .NET's own nesting convention, so <c>Ollama__ApiKey</c> in the file
/// addresses <c>Ollama:ApiKey</c> in configuration and the identical name works
/// as a real environment variable in a container.
///
/// A whole configuration package would be more than this needs. The one rule that
/// matters is that the file must not override a real environment variable: this
/// runs last, so anything it added would otherwise win, and a stray local .env
/// left in a deployed image would quietly replace the deployment's own settings.
/// </summary>
internal static class DotEnv
{
    public static void Load(string path, IConfigurationManager configuration)
    {
        if (!File.Exists(path)) return;

        var values = new Dictionary<string, string?>();

        foreach (var line in File.ReadAllLines(path))
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0 || trimmed.StartsWith('#')) continue;

            var separator = trimmed.IndexOf('=');
            if (separator <= 0) continue;

            var key = trimmed[..separator].Trim().Replace("__", ":");
            var value = trimmed[(separator + 1)..].Trim().Trim('"', '\'');
            if (key.Length == 0) continue;

            // Only fill gaps. Anything already configured came from a source with
            // a stronger claim than a file sitting in the working directory.
            if (!string.IsNullOrEmpty(configuration[key])) continue;

            values[key] = value;
        }

        if (values.Count > 0) configuration.AddInMemoryCollection(values);
    }
}

/// <summary>Exposed so an integration test project can build this host.</summary>
public partial class Program { }
