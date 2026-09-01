using System.Text;
using JobPortal.Api.Contracts;
using JobPortal.Api.Controllers;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Jobs;
using JobPortal.Api.Services.Vectors;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace JobPortal.Api.Tests;

/// <summary>
/// Bulk JD upload.
///
/// The behaviour worth locking is not the happy path — it is that ONE bad document
/// cannot sink the batch. A recruiter dropping thirty JDs will have a scanned PDF or
/// two in there, and losing the twenty-eight that parsed because of them would make
/// the feature worse than uploading one at a time.
/// </summary>
public class BulkJobUploadTests
{
    // --- Fakes: only the dependencies UploadBulk actually touches -----------------
    private sealed class FakeExtractor : IDocumentTextExtractor
    {
        public bool IsSupported(string filename) =>
            filename.EndsWith(".pdf") || filename.EndsWith(".docx") ||
            filename.EndsWith(".txt") || filename.EndsWith(".md");

        public string Extract(Stream stream, string filename) => "text";
        public string Extract(string path) => "text";
    }

    private sealed class FakeJobs : IJobService
    {
        /// <summary>Filenames that should blow up, to simulate an unreadable document.</summary>
        public HashSet<string> Unreadable { get; } = new();
        public int Created { get; private set; }

        public Task<PortalJob> CreateFromDocumentAsync(
            Stream file, string filename, string? storedPath, int? sourceJdId, CancellationToken ct = default)
        {
            if (Unreadable.Contains(filename))
                throw new InvalidOperationException("No readable text in the document.");

            Created++;
            return Task.FromResult(new PortalJob
            {
                Id = Created,
                Title = Path.GetFileNameWithoutExtension(filename),
                IsPublished = true,
            });
        }

        public Task<PortalJob> CreateFromRequestAsync(CreateJobRequest r, CancellationToken ct = default) =>
            throw new NotImplementedException();
        public Task<bool> IndexAsync(PortalJob job, CancellationToken ct = default) =>
            throw new NotImplementedException();
        public Task<int> ReindexAllAsync(bool force, CancellationToken ct = default) =>
            throw new NotImplementedException();
        public Task<ImportResultDto> ImportFromPythonAsync(CancellationToken ct = default) =>
            throw new NotImplementedException();
        public Task<IndexStatusDto> GetIndexStatusAsync(CancellationToken ct = default) =>
            throw new NotImplementedException();
    }

    private sealed class FakeVectors : IVectorStore
    {
        public Task<IReadOnlyDictionary<int, string>> GetIndexedHashesAsync(
            string model, CancellationToken ct = default) =>
            Task.FromResult<IReadOnlyDictionary<int, string>>(new Dictionary<int, string>());

        public Task UpsertAsync(int jobId, string model, int dimension, float[] vector,
            string textHash, CancellationToken ct = default) => Task.CompletedTask;
        public Task<IReadOnlyList<VectorHit>> SearchAsync(float[] query, string model, int topK,
            double minScore = 0, IReadOnlySet<int>? restrictTo = null, CancellationToken ct = default) =>
            Task.FromResult<IReadOnlyList<VectorHit>>(Array.Empty<VectorHit>());
        public Task<int> CountAsync(string model, CancellationToken ct = default) => Task.FromResult(0);
        public Task<IReadOnlyDictionary<string, int>> CountsByModelAsync(CancellationToken ct = default) =>
            Task.FromResult<IReadOnlyDictionary<string, int>>(new Dictionary<string, int>());
        public Task RemoveAsync(int jobId, CancellationToken ct = default) => Task.CompletedTask;
    }

    private sealed class FakeEmbeddings : IEmbeddingService
    {
        public string PreferredModelId => "test-model";
        public bool SemanticAvailable => false;
        public Task<EmbeddingBatch> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default) =>
            throw new NotImplementedException();
        public Task<EmbeddingBatch> EmbedAsync(string text, CancellationToken ct = default) =>
            throw new NotImplementedException();
    }

    private static IFormFile File(string name, string content = "Job Title: Engineer")
    {
        var bytes = Encoding.UTF8.GetBytes(content);
        return new FormFile(new MemoryStream(bytes), 0, bytes.Length, "files", name);
    }

    private static IFormFileCollection Collection(params IFormFile[] files)
    {
        var collection = new FormFileCollection();
        collection.AddRange(files);
        return collection;
    }

    /// <param name="jobs">Passed in when the test needs to inspect or script it.</param>
    private static JobsController Controller(FakeJobs? jobs = null) =>
        // _db and _suggestions are genuinely unused by UploadBulk; supplying real ones
        // would only obscure which dependencies the endpoint actually has.
        new(null!, jobs ?? new FakeJobs(), null!, new FakeVectors(), new FakeEmbeddings(),
            new FakeExtractor(), NullLogger<JobsController>.Instance);

    private static JobBulkUploadResultDto Payload(ActionResult<ApiResponse<JobBulkUploadResultDto>> result)
    {
        var ok = Assert.IsType<OkObjectResult>(result.Result);
        var body = Assert.IsType<ApiResponse<JobBulkUploadResultDto>>(ok.Value);
        Assert.NotNull(body.Data);
        return body.Data!;
    }

    // --- Tests ---------------------------------------------------------------------
    [Fact]
    public async Task Posts_every_readable_document()
    {
        var result = await Controller().UploadBulk(
            Collection(File("a.pdf"), File("b.docx"), File("c.md")), default);

        var payload = Payload(result);
        Assert.Equal(3, payload.Total);
        Assert.Equal(3, payload.Succeeded);
        Assert.Equal(0, payload.Failed);
        Assert.All(payload.Items, item => Assert.True(item.Success));
    }

    [Fact]
    public async Task One_unreadable_document_does_not_sink_the_batch()
    {
        var jobs = new FakeJobs();
        jobs.Unreadable.Add("broken.pdf");

        var payload = Payload(await Controller(jobs).UploadBulk(
            Collection(File("good1.pdf"), File("broken.pdf"), File("good2.pdf")), default));

        Assert.Equal(2, payload.Succeeded);
        Assert.Equal(1, payload.Failed);
        Assert.Equal(2, jobs.Created);
    }

    [Fact]
    public async Task A_failure_names_the_file_and_says_why()
    {
        var jobs = new FakeJobs();
        jobs.Unreadable.Add("scan.pdf");

        var payload = Payload(await Controller(jobs).UploadBulk(
            Collection(File("ok.pdf"), File("scan.pdf")), default));

        var failure = Assert.Single(payload.Items, i => !i.Success);
        Assert.Equal("scan.pdf", failure.Filename);
        Assert.Contains("readable", failure.Error!, StringComparison.OrdinalIgnoreCase);
        Assert.Null(failure.Job);
    }

    [Fact]
    public async Task Unsupported_extensions_are_rejected_per_file_not_per_batch()
    {
        var payload = Payload(await Controller().UploadBulk(
            Collection(File("cv.pdf"), File("notes.xlsx")), default));

        Assert.Equal(1, payload.Succeeded);
        Assert.Equal(1, payload.Failed);
        Assert.Contains(".xlsx", payload.Items.Single(i => !i.Success).Error!);
    }

    [Fact]
    public async Task Empty_files_are_reported_rather_than_parsed()
    {
        var payload = Payload(await Controller().UploadBulk(
            Collection(File("empty.pdf", ""), File("real.pdf")), default));

        Assert.Equal(1, payload.Succeeded);
        Assert.Contains("empty", payload.Items.Single(i => !i.Success).Error!,
            StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task An_empty_request_is_a_bad_request()
    {
        var result = await Controller().UploadBulk(Collection(), default);
        Assert.IsType<BadRequestObjectResult>(result.Result);
    }

    [Fact]
    public async Task A_batch_over_the_cap_is_refused_before_any_work_happens()
    {
        var jobs = new FakeJobs();
        var tooMany = Enumerable.Range(0, JobsController.MaxBulkFiles + 1)
            .Select(i => File($"jd{i}.pdf")).ToArray();

        var result = await Controller(jobs).UploadBulk(Collection(tooMany), default);

        Assert.IsType<BadRequestObjectResult>(result.Result);
        Assert.Equal(0, jobs.Created);
    }

    [Fact]
    public async Task The_cap_itself_is_accepted()
    {
        var files = Enumerable.Range(0, JobsController.MaxBulkFiles)
            .Select(i => File($"jd{i}.pdf")).ToArray();

        var payload = Payload(await Controller().UploadBulk(Collection(files), default));
        Assert.Equal(JobsController.MaxBulkFiles, payload.Succeeded);
    }
}
