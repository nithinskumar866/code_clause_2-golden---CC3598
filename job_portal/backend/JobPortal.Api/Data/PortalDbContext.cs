using Microsoft.EntityFrameworkCore;

namespace JobPortal.Api.Data;

public class PortalDbContext : DbContext
{
    public PortalDbContext(DbContextOptions<PortalDbContext> options) : base(options) { }

    // Portal-owned
    public DbSet<PortalJob> Jobs => Set<PortalJob>();
    public DbSet<PortalJobVector> JobVectors => Set<PortalJobVector>();
    public DbSet<PortalResume> Resumes => Set<PortalResume>();
    public DbSet<PortalChatSession> ChatSessions => Set<PortalChatSession>();
    public DbSet<PortalChatMessage> ChatMessages => Set<PortalChatMessage>();
    public DbSet<PortalApplication> Applications => Set<PortalApplication>();
    public DbSet<PortalSkillVerdictCache> SkillVerdictCache => Set<PortalSkillVerdictCache>();

    // Python-owned, read-only
    public DbSet<PythonJobDescription> PythonJobDescriptions => Set<PythonJobDescription>();
    public DbSet<PythonResume> PythonResumes => Set<PythonResume>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        // The Python platform owns these two. Marking them no-key-less but
        // read-only in intent; EF must never emit DDL or writes for them.
        b.Entity<PythonJobDescription>().ToTable("job_descriptions").HasKey(x => x.Id);
        b.Entity<PythonResume>().ToTable("resumes").HasKey(x => x.Id);

        b.Entity<PortalJob>(e =>
        {
            e.HasIndex(x => x.SourceJobDescriptionId);
            e.HasIndex(x => x.IsPublished);
            e.HasMany(x => x.Vectors)
             .WithOne(v => v.Job!)
             .HasForeignKey(v => v.JobId)
             .OnDelete(DeleteBehavior.Cascade);
        });

        b.Entity<PortalJobVector>(e =>
        {
            // One vector per (job, model) — the dimension is a property of the
            // model, so it is not part of the identity.
            e.HasIndex(x => new { x.JobId, x.Model }).IsUnique();
            e.HasIndex(x => x.Model);
        });

        b.Entity<PortalResume>(e =>
        {
            e.HasIndex(x => x.ContentHash);
        });

        b.Entity<PortalChatSession>(e =>
        {
            e.HasMany(x => x.Messages)
             .WithOne(m => m.Session!)
             .HasForeignKey(m => m.SessionId)
             .OnDelete(DeleteBehavior.Cascade);
            e.HasOne(x => x.Resume)
             .WithMany()
             .HasForeignKey(x => x.ResumeId)
             .OnDelete(DeleteBehavior.SetNull);
        });

        b.Entity<PortalChatMessage>(e =>
        {
            e.HasIndex(x => x.SessionId);
        });

        b.Entity<PortalApplication>(e =>
        {
            // Keyed on the document, not the row: a candidate who re-uploads the
            // same CV gets a new portal_resumes row, and must still be unable to
            // apply to the same posting twice.
            e.HasIndex(x => new { x.JobId, x.ResumeContentHash }).IsUnique();
            e.HasIndex(x => x.JobId);
            e.HasIndex(x => x.ResumeId);
            e.HasIndex(x => x.Status);

            e.HasOne(x => x.Job)
             .WithMany()
             .HasForeignKey(x => x.JobId)
             .OnDelete(DeleteBehavior.Cascade);

            e.HasOne(x => x.Resume)
             .WithMany()
             .HasForeignKey(x => x.ResumeId)
             .OnDelete(DeleteBehavior.Cascade);
        });

        b.Entity<PortalSkillVerdictCache>(e =>
        {
            // Unique per (job, candidate skill set, required skill)
            e.HasIndex(x => new { x.JobId, x.CandidateSkillSetHash, x.RequiredSkill }).IsUnique();
            e.HasIndex(x => x.JobId);
            e.HasIndex(x => x.PromptVersion);
        });
    }
}
