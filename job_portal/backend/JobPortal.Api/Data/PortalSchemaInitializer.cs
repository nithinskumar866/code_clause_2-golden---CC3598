using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace JobPortal.Api.Data;

/// <summary>
/// Creates the portal's own tables inside a SQLite file that the Python platform
/// also owns.
///
/// Why raw DDL instead of <c>EnsureCreated</c> or EF migrations: <c>EnsureCreated</c>
/// is all-or-nothing — it does nothing at all once the file contains any table, so
/// on a database SQLAlchemy created first the portal's tables would silently never
/// appear. EF migrations would work, but their history table and model snapshot
/// would assert ownership over a schema the Python side actually owns, and a
/// stray <c>dotnet ef database update</c> could then try to "fix" SQLAlchemy's
/// tables. Idempotent CREATE TABLE IF NOT EXISTS touches only what is ours.
/// </summary>
public static class PortalSchemaInitializer
{
    private static readonly string[] Ddl =
    {
        """
        CREATE TABLE IF NOT EXISTS portal_jobs (
            Id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            SourceJobDescriptionId  INTEGER NULL,
            SourceFilePath          TEXT NULL,
            Title                   TEXT NOT NULL DEFAULT '',
            Company                 TEXT NOT NULL DEFAULT '',
            Location                TEXT NOT NULL DEFAULT '',
            WorkMode                TEXT NOT NULL DEFAULT 'Unspecified',
            EmploymentType          TEXT NOT NULL DEFAULT 'Unspecified',
            SeniorityLevel          TEXT NOT NULL DEFAULT 'Unspecified',
            MinYearsExperience      REAL NULL,
            MaxYearsExperience      REAL NULL,
            SalaryMin               TEXT NULL,
            SalaryMax               TEXT NULL,
            SalaryCurrency          TEXT NULL,
            RequiredSkills          TEXT NOT NULL DEFAULT '',
            PreferredSkills         TEXT NOT NULL DEFAULT '',
            Responsibilities        TEXT NOT NULL DEFAULT '',
            Qualifications          TEXT NOT NULL DEFAULT '',
            Summary                 TEXT NOT NULL DEFAULT '',
            RawText                 TEXT NOT NULL DEFAULT '',
            EmbeddedText            TEXT NOT NULL DEFAULT '',
            ExtractionMode          TEXT NOT NULL DEFAULT 'deterministic',
            ApplyUrl                TEXT NULL,
            IsPublished             INTEGER NOT NULL DEFAULT 1,
            CreatedAt               TEXT NOT NULL,
            UpdatedAt               TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS IX_portal_jobs_SourceJobDescriptionId ON portal_jobs (SourceJobDescriptionId)",
        "CREATE INDEX IF NOT EXISTS IX_portal_jobs_IsPublished ON portal_jobs (IsPublished)",

        """
        CREATE TABLE IF NOT EXISTS portal_job_vectors (
            Id         INTEGER PRIMARY KEY AUTOINCREMENT,
            JobId      INTEGER NOT NULL,
            Model      TEXT NOT NULL,
            Dimension  INTEGER NOT NULL,
            Vector     BLOB NOT NULL,
            TextHash   TEXT NOT NULL DEFAULT '',
            CreatedAt  TEXT NOT NULL,
            FOREIGN KEY (JobId) REFERENCES portal_jobs (Id) ON DELETE CASCADE
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS IX_portal_job_vectors_JobId_Model ON portal_job_vectors (JobId, Model)",
        "CREATE INDEX IF NOT EXISTS IX_portal_job_vectors_Model ON portal_job_vectors (Model)",

        """
        CREATE TABLE IF NOT EXISTS portal_resumes (
            Id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            Filename            TEXT NOT NULL DEFAULT '',
            StoredPath          TEXT NOT NULL DEFAULT '',
            ContentHash         TEXT NOT NULL DEFAULT '',
            RawText             TEXT NOT NULL DEFAULT '',
            CandidateName       TEXT NOT NULL DEFAULT '',
            Email               TEXT NOT NULL DEFAULT '',
            Phone               TEXT NOT NULL DEFAULT '',
            Location            TEXT NOT NULL DEFAULT '',
            CurrentTitle        TEXT NOT NULL DEFAULT '',
            YearsExperience     REAL NULL,
            Skills              TEXT NOT NULL DEFAULT '',
            Titles              TEXT NOT NULL DEFAULT '',
            Education           TEXT NOT NULL DEFAULT '',
            Summary             TEXT NOT NULL DEFAULT '',
            ExtractionMode      TEXT NOT NULL DEFAULT 'deterministic',
            EmbeddingModel      TEXT NOT NULL DEFAULT '',
            EmbeddingDimension  INTEGER NOT NULL DEFAULT 0,
            Embedding           BLOB NULL,
            CreatedAt           TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS IX_portal_resumes_ContentHash ON portal_resumes (ContentHash)",

        """
        CREATE TABLE IF NOT EXISTS portal_chat_sessions (
            Id              TEXT NOT NULL PRIMARY KEY,
            ResumeId        INTEGER NULL,
            FiltersJson     TEXT NOT NULL DEFAULT '{}',
            LastMatchesJson TEXT NOT NULL DEFAULT '[]',
            FocusJobId      INTEGER NULL,
            LastFactsJson   TEXT NOT NULL DEFAULT '{}',
            FilterHistoryJson TEXT NOT NULL DEFAULT '[]',
            CreatedAt       TEXT NOT NULL,
            UpdatedAt       TEXT NOT NULL,
            FOREIGN KEY (ResumeId) REFERENCES portal_resumes (Id) ON DELETE SET NULL
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS portal_chat_messages (
            Id        INTEGER PRIMARY KEY AUTOINCREMENT,
            SessionId TEXT NOT NULL,
            Role      TEXT NOT NULL DEFAULT 'user',
            Content   TEXT NOT NULL DEFAULT '',
            CreatedAt TEXT NOT NULL,
            FOREIGN KEY (SessionId) REFERENCES portal_chat_sessions (Id) ON DELETE CASCADE
        )
        """,
        "CREATE INDEX IF NOT EXISTS IX_portal_chat_messages_SessionId ON portal_chat_messages (SessionId)",

        """
        CREATE TABLE IF NOT EXISTS portal_applications (
            Id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            JobId              INTEGER NOT NULL,
            ResumeId           INTEGER NOT NULL,
            ResumeContentHash  TEXT NOT NULL DEFAULT '',
            Status             TEXT NOT NULL DEFAULT 'Submitted',
            CoverLetter        TEXT NOT NULL DEFAULT '',
            LetterMode         TEXT NOT NULL DEFAULT 'deterministic',
            FitScore           REAL NOT NULL DEFAULT 0,
            FitBand            TEXT NOT NULL DEFAULT '',
            SemanticScore      REAL NOT NULL DEFAULT 0,
            SkillScore         REAL NOT NULL DEFAULT 0,
            TitleScore         REAL NOT NULL DEFAULT 0,
            ExperienceScore    REAL NOT NULL DEFAULT 0,
            SkillsJson         TEXT NOT NULL DEFAULT '[]',
            StrengthsJson      TEXT NOT NULL DEFAULT '[]',
            GapsJson           TEXT NOT NULL DEFAULT '[]',
            RecruiterNote      TEXT NOT NULL DEFAULT '',
            SemanticMatching   INTEGER NOT NULL DEFAULT 0,
            EmbeddingModel     TEXT NOT NULL DEFAULT '',
            ModelVersion       TEXT NOT NULL DEFAULT '',
            PromptVersion      TEXT NOT NULL DEFAULT '',
            SkillVerdictCacheKeysJson TEXT NOT NULL DEFAULT '[]',
            AcceptedAnalysisId INTEGER NULL,
            AcceptedAt         TEXT NULL,
            CreatedAt          TEXT NOT NULL,
            UpdatedAt          TEXT NOT NULL,
            FOREIGN KEY (JobId) REFERENCES portal_jobs (Id) ON DELETE CASCADE,
            FOREIGN KEY (ResumeId) REFERENCES portal_resumes (Id) ON DELETE CASCADE
        )
        """,
        // Applying twice to one posting is the failure mode of bulk apply, and a
        // service-level check cannot be trusted with it: the same person re-uploads
        // their CV, gets a new portal_resumes row, and applies again in good faith.
        // Keying on the CONTENT HASH rather than ResumeId makes the second attempt
        // impossible at the database, whichever row it came from.
        "CREATE UNIQUE INDEX IF NOT EXISTS IX_portal_applications_Job_Resume " +
        "ON portal_applications (JobId, ResumeContentHash)",
        "CREATE INDEX IF NOT EXISTS IX_portal_applications_JobId ON portal_applications (JobId)",
        "CREATE INDEX IF NOT EXISTS IX_portal_applications_ResumeId ON portal_applications (ResumeId)",
        "CREATE INDEX IF NOT EXISTS IX_portal_applications_Status ON portal_applications (Status)",
    };

    public static async Task InitializeAsync(PortalDbContext db, ILogger logger, CancellationToken ct = default)
    {
        // Creates the file if the Python side has never run. Harmless if it has.
        var connection = (SqliteConnection)db.Database.GetDbConnection();
        await connection.OpenAsync(ct);
        try
        {
            foreach (var statement in Ddl)
            {
                // A plain ADO command rather than ExecuteSqlRaw: EF treats raw SQL
                // as a composite format string and chokes on the braces and quoting
                // that DDL is full of. There is nothing to parameterise here anyway.
                await using var cmd = connection.CreateCommand();
                cmd.CommandText = statement;
                await cmd.ExecuteNonQueryAsync(ct);
            }

            await EnsureColumnsAsync(connection, logger, ct);
        }
        finally
        {
            await connection.CloseAsync();
        }

        logger.LogInformation("Portal schema ready in {Db}", db.Database.GetDbConnection().DataSource);

        foreach (var table in new[] { "job_descriptions", "resumes" })
        {
            if (!await TableExistsAsync(db, table, ct))
            {
                logger.LogWarning(
                    "Python table '{Table}' is absent from the shared database. The portal runs " +
                    "fine without it; importing recruiter-side {Table} will return nothing until " +
                    "the FastAPI backend has created it.", table, table);
            }
        }
    }

    /// <summary>
    /// Columns added to a portal table AFTER it first shipped.
    ///
    /// <c>CREATE TABLE IF NOT EXISTS</c> creates; it does not alter. On any
    /// database where the table already exists — which is every database that has
    /// run this build's predecessor — a new property on an entity produces
    /// "no such column" on the first query that touches it, and the feature is
    /// dead until someone deletes the file. That is the same all-or-nothing trap
    /// this class avoids <c>EnsureCreated</c> for, one level down.
    ///
    /// Every entry must be nullable or carry a DEFAULT: SQLite cannot add a NOT
    /// NULL column to a table that already has rows without one.
    /// </summary>
    private static readonly (string Table, string Column, string Definition)[] AdditiveColumns =
    {
        ("portal_applications", "ModelVersion", "TEXT NOT NULL DEFAULT ''"),
        ("portal_applications", "PromptVersion", "TEXT NOT NULL DEFAULT ''"),
        ("portal_applications", "SkillVerdictCacheKeysJson", "TEXT NOT NULL DEFAULT '[]'"),

        // Conversation memory. Follow-up turns need to know what was just said, what
        // is on screen, and what the filters used to be — none of which the session
        // held. "Is it remote?" and "go back" are unanswerable without these.
        ("portal_chat_sessions", "FocusJobId", "INTEGER NULL"),
        ("portal_chat_sessions", "LastFactsJson", "TEXT NOT NULL DEFAULT '{}'"),
        ("portal_chat_sessions", "FilterHistoryJson", "TEXT NOT NULL DEFAULT '[]'"),

        // Which scorer the conversation is set to. Defaults to the arithmetic one,
        // so an existing session keeps behaving exactly as it did before the choice
        // existed.
        ("portal_chat_sessions", "ScoringMode", "TEXT NOT NULL DEFAULT 'computed'"),
    };

    /// <summary>
    /// Brings existing portal tables up to the current shape, additively.
    ///
    /// Only ever ADDs, and only to <c>portal_</c> tables. Nothing here drops,
    /// renames or retypes anything, and nothing touches a table the Python side
    /// owns — an additive column is the one schema change that cannot break a
    /// reader that has not heard of it.
    /// </summary>
    private static async Task EnsureColumnsAsync(
        SqliteConnection connection, ILogger logger, CancellationToken ct)
    {
        foreach (var group in AdditiveColumns.GroupBy(c => c.Table))
        {
            if (!group.Key.StartsWith("portal_", StringComparison.Ordinal))
                throw new InvalidOperationException($"Refusing to alter non-portal table '{group.Key}'.");

            var existing = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            await using (var probe = connection.CreateCommand())
            {
                probe.CommandText = $"PRAGMA table_info({group.Key})";
                await using var reader = await probe.ExecuteReaderAsync(ct);
                while (await reader.ReadAsync(ct)) existing.Add(reader.GetString(1));
            }

            // No rows at all means the table does not exist. The DDL above owns
            // creating it; adding columns to nothing is not this method's job.
            if (existing.Count == 0) continue;

            foreach (var (table, column, definition) in group)
            {
                if (existing.Contains(column)) continue;

                await using var alter = connection.CreateCommand();
                alter.CommandText = $"ALTER TABLE {table} ADD COLUMN {column} {definition}";
                await alter.ExecuteNonQueryAsync(ct);

                logger.LogInformation("Added missing column {Table}.{Column}", table, column);
            }
        }
    }

    public static async Task<bool> TableExistsAsync(PortalDbContext db, string table, CancellationToken ct = default)
    {
        var connection = (SqliteConnection)db.Database.GetDbConnection();
        var opened = connection.State != System.Data.ConnectionState.Open;
        if (opened) await connection.OpenAsync(ct);
        try
        {
            await using var cmd = connection.CreateCommand();
            cmd.CommandText = "SELECT 1 FROM sqlite_master WHERE type='table' AND name = $name LIMIT 1";
            cmd.Parameters.AddWithValue("$name", table);
            return await cmd.ExecuteScalarAsync(ct) is not null;
        }
        finally
        {
            if (opened) await connection.CloseAsync();
        }
    }
}
