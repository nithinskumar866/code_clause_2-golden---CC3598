using Microsoft.EntityFrameworkCore;

namespace JobPortal.Api.Services.Rag.Data;

/// <summary>
/// One chunk of one document — the text, and where in the document it came from.
///
/// Deliberately model-INDEPENDENT. Parsing and chunking are expensive and depend
/// only on the document; embedding is cheap by comparison and depends on the
/// model. Keeping them apart means adding or swapping an embedding model
/// re-embeds this table rather than re-parsing a hundred thousand PDFs.
///
/// <see cref="ChunkId"/> is the shared key: Qdrant stores it as the point id, so
/// a vector hit maps back to text with no second lookup and no id translation.
/// </summary>
public class RagChunk
{
    public long ChunkId { get; set; }

    /// <summary>The id of the row in the portal's own tables — a job or a résumé.</summary>
    public int ParentId { get; set; }

    /// <summary>See <see cref="RagParentTypes"/>. Namespaces <see cref="ParentId"/>,
    /// which is only unique within its own table.</summary>
    public string ParentType { get; set; } = "";

    /// <summary>
    /// Which part of the document this came from — Experience, Skills, Projects.
    ///
    /// The single most useful piece of metadata here: it is what separates a skill
    /// someone demonstrated from one they listed, and it is what the evidence card
    /// shows beside every quote.
    /// </summary>
    public string Section { get; set; } = "";

    /// <summary>Position within the document, so a résumé can be read back in order.</summary>
    public int Ordinal { get; set; }

    public int Page { get; set; }

    public string Text { get; set; } = "";

    /// <summary>SHA-256 of the source document's text. Lets a re-ingest skip
    /// documents whose content has not moved.</summary>
    public string SourceHash { get; set; } = "";

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

public static class RagParentTypes
{
    public const string Resume = "resume";
    public const string Job = "job";
}

/// <summary>
/// What has been ingested, how far, and what went wrong.
///
/// Separate from the chunks so a failed or half-finished document is visible
/// without inferring it from a chunk count. Ingestion is resumable because this
/// row survives a restart; without it, a crash at 60,000 documents means starting
/// at zero.
/// </summary>
public class RagIngestState
{
    public long Id { get; set; }

    public int ParentId { get; set; }

    public string ParentType { get; set; } = "";

    /// <summary>See <see cref="RagIngestStatuses"/>.</summary>
    public string Status { get; set; } = RagIngestStatuses.Pending;

    /// <summary>The document text's hash at the time it was ingested. A document
    /// whose hash still matches needs no work, which is what makes a re-run free.</summary>
    public string SourceHash { get; set; } = "";

    public int ChunkCount { get; set; }

    public int Attempts { get; set; }

    /// <summary>Why the last attempt failed. Kept so a parked document can be
    /// diagnosed without reproducing the failure.</summary>
    public string? LastError { get; set; }

    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}

public static class RagIngestStatuses
{
    public const string Pending = "pending";
    public const string Indexed = "indexed";
    /// <summary>Exhausted its attempts. Never retried automatically.</summary>
    public const string Failed = "failed";
}

/// <summary>
/// The RAG mode's own database.
///
/// Postgres, and entirely separate from <c>PortalDbContext</c>: the existing
/// scoring modes stay on SQLite and must not move as a side effect of this
/// feature. Nothing dual-writes — this side treats the portal's tables as a
/// read-only source of truth and stores only what it derives from them.
/// </summary>
public class RagDbContext : DbContext
{
    public RagDbContext(DbContextOptions<RagDbContext> options) : base(options) { }

    public DbSet<RagChunk> Chunks => Set<RagChunk>();
    public DbSet<RagIngestState> IngestState => Set<RagIngestState>();

    protected override void OnModelCreating(ModelBuilder model)
    {
        model.Entity<RagChunk>(e =>
        {
            e.ToTable("rag_chunks");
            e.HasKey(c => c.ChunkId);
            e.Property(c => c.ParentType).HasMaxLength(16).IsRequired();
            e.Property(c => c.Section).HasMaxLength(64).IsRequired();
            e.Property(c => c.SourceHash).HasMaxLength(64).IsRequired();
            e.Property(c => c.Text).IsRequired();

            // Every read of this table is "give me the chunks of document X" —
            // deleting and re-chunking one document, or reading one résumé back in
            // order. Without this index that is a sequential scan of 1.5M rows.
            e.HasIndex(c => new { c.ParentType, c.ParentId }).HasDatabaseName("ix_rag_chunks_parent");
        });

        model.Entity<RagIngestState>(e =>
        {
            e.ToTable("rag_ingest_state");
            e.HasKey(s => s.Id);
            e.Property(s => s.ParentType).HasMaxLength(16).IsRequired();
            e.Property(s => s.Status).HasMaxLength(16).IsRequired();
            e.Property(s => s.SourceHash).HasMaxLength(64).IsRequired();

            // One row per document, enforced by the database rather than by the
            // worker remembering to check. Two workers racing the same document
            // then collide on insert instead of both indexing it.
            e.HasIndex(s => new { s.ParentType, s.ParentId })
                .IsUnique()
                .HasDatabaseName("ux_rag_ingest_parent");

            // The worker's own query: what is still outstanding.
            e.HasIndex(s => s.Status).HasDatabaseName("ix_rag_ingest_status");
        });
    }
}
