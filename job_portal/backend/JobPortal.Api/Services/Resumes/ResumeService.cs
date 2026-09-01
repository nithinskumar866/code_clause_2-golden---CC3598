using System.Security.Cryptography;
using JobPortal.Api.Data;
using JobPortal.Api.Options;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Vectors;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Resumes;

public interface IResumeService
{
    Task<PortalResume> IngestAsync(Stream file, string filename, CancellationToken ct = default);
    Task<PortalResume?> GetAsync(int id, CancellationToken ct = default);
}

public class ResumeService : IResumeService
{
    private readonly PortalDbContext _db;
    private readonly IDocumentTextExtractor _extractor;
    private readonly IResumeProfileService _profiles;
    private readonly IEmbeddingService _embeddings;
    private readonly PortalOptions _portal;
    private readonly ILogger<ResumeService> _logger;

    public ResumeService(
        PortalDbContext db,
        IDocumentTextExtractor extractor,
        IResumeProfileService profiles,
        IEmbeddingService embeddings,
        IOptions<PortalOptions> portal,
        ILogger<ResumeService> logger)
    {
        _db = db;
        _extractor = extractor;
        _profiles = profiles;
        _embeddings = embeddings;
        _portal = portal.Value;
        _logger = logger;
    }

    public Task<PortalResume?> GetAsync(int id, CancellationToken ct = default) =>
        _db.Resumes.FirstOrDefaultAsync(r => r.Id == id, ct);

    public async Task<PortalResume> IngestAsync(Stream file, string filename, CancellationToken ct = default)
    {
        if (!_extractor.IsSupported(filename))
        {
            throw new NotSupportedException(
                $"'{Path.GetExtension(filename)}' is not supported. Upload a PDF, DOCX, TXT or MD file.");
        }

        var buffer = new MemoryStream();
        await file.CopyToAsync(buffer, ct);
        buffer.Position = 0;

        var hash = Convert.ToHexString(SHA256.HashData(buffer.ToArray())).ToLowerInvariant();
        buffer.Position = 0;

        // The same file re-uploaded is the same candidate. Returning the existing
        // row keeps a refreshed browser tab from creating a second profile and a
        // second set of matches for one person.
        var existing = await _db.Resumes.FirstOrDefaultAsync(r => r.ContentHash == hash, ct);
        if (existing is not null)
        {
            _logger.LogInformation("Resume '{Filename}' already ingested as {Id}", filename, existing.Id);
            return existing;
        }

        var rawText = _extractor.Extract(buffer, filename);
        var profile = await _profiles.ExtractAsync(rawText, filename, ct);

        var resume = new PortalResume
        {
            Filename = filename,
            ContentHash = hash,
            RawText = rawText,
            CandidateName = profile.CandidateName,
            Email = profile.Email,
            Phone = profile.Phone,
            Location = profile.Location,
            CurrentTitle = profile.CurrentTitle,
            YearsExperience = profile.YearsExperience,
            Skills = TextStructure.JoinLines(profile.Skills),
            Titles = TextStructure.JoinLines(profile.Titles),
            Education = TextStructure.JoinLines(profile.Education),
            Summary = profile.Summary,
            ExtractionMode = profile.Mode,
        };

        resume.StoredPath = await StoreAsync(buffer, filename, hash, ct);

        _db.Resumes.Add(resume);
        await _db.SaveChangesAsync(ct);

        await EmbedAsync(resume, ct);
        return resume;
    }

    private async Task<string> StoreAsync(MemoryStream buffer, string filename, string hash, CancellationToken ct)
    {
        Directory.CreateDirectory(_portal.ResumeUploadDir);

        // Content hash in the name, so two candidates who both uploaded
        // "resume.pdf" do not overwrite each other.
        var safeName = Path.GetFileName(filename);
        var path = Path.Combine(_portal.ResumeUploadDir, $"{hash[..12]}_{safeName}");

        buffer.Position = 0;
        await using (var target = File.Create(path))
        {
            await buffer.CopyToAsync(target, ct);
        }
        buffer.Position = 0;
        return path;
    }

    private async Task EmbedAsync(PortalResume resume, CancellationToken ct)
    {
        var text = _profiles.BuildEmbeddingText(resume);
        var batch = await _embeddings.EmbedAsync(text, ct);
        if (batch.Vectors.Count == 0) return;

        resume.EmbeddingModel = batch.ModelId;
        resume.EmbeddingDimension = batch.Dimension;
        resume.Embedding = VectorMath.ToBytes(VectorMath.Normalize(batch.Vectors[0]));

        await _db.SaveChangesAsync(ct);
    }
}
