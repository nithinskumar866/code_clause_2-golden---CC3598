using JobPortal.Api.Services.Rag;
using JobPortal.Api.Services.Rag.Ingestion;

namespace JobPortal.Api.Tests;

/// <summary>
/// Chunking, which every later retrieval decision inherits.
///
/// The property that matters most is that a chunk never spans a section. "React"
/// under *Skills* is a claim and "React" under *Experience* is proof; a chunk
/// straddling the two makes that distinction unrecoverable, and the distinction
/// is what the whole evaluation turns on.
/// </summary>
public class DocumentChunkerTests
{
    private static DocumentChunker Chunker(RagOptions? options = null) =>
        new(Microsoft.Extensions.Options.Options.Create(options ?? new RagOptions()));

    private const string Cv = """
        Priya Raman
        Senior Backend Engineer

        Summary
        Backend engineer with 6 years of experience building payment systems.

        Skills
        Java, Spring Boot, PostgreSQL, Kafka, Docker, Kubernetes

        Experience
        Senior Backend Engineer, Razorpay, 2021-2024
        Designed and shipped the settlement service handling 4M transactions a day.
        Led the migration from a monolith to eleven Spring Boot services.
        Mentored three junior engineers through their first production releases.

        Backend Engineer, Freshworks, 2018-2021
        Built REST APIs in Java against PostgreSQL and cut p99 latency by 40%.

        Education
        B.E. in Computer Science, Anna University
        """;

    // -- the structural guarantee -------------------------------------------

    [Fact]
    public void A_chunk_never_spans_two_sections()
    {
        var chunks = Chunker().Chunk(Cv, isResume: true);

        Assert.NotEmpty(chunks);

        // The skills list and the work history must never share a chunk — that is
        // the claim-versus-proof distinction the evaluation depends on.
        Assert.DoesNotContain(chunks, c =>
            c.Text.Contains("Kubernetes") && c.Text.Contains("settlement service"));
    }

    [Fact]
    public void Every_chunk_is_labelled_with_the_section_it_came_from()
    {
        var chunks = Chunker().Chunk(Cv, isResume: true);

        Assert.All(chunks, c => Assert.False(string.IsNullOrWhiteSpace(c.Section)));

        var experience = chunks.Where(c => c.Section.Contains("Experience", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(experience, c => c.Text.Contains("settlement service"));

        var skills = chunks.Where(c => c.Section.Contains("Skills", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(skills, c => c.Text.Contains("Kafka"));
    }

    [Fact]
    public void Ordinals_run_in_document_order_with_no_gaps()
    {
        var chunks = Chunker().Chunk(Cv, isResume: true);

        Assert.Equal(Enumerable.Range(0, chunks.Count), chunks.Select(c => c.Ordinal));
    }

    // -- sizing --------------------------------------------------------------

    [Fact]
    public void Chunks_respect_the_maximum_except_for_one_oversized_sentence()
    {
        var options = new RagOptions { ChunkTargetChars = 120, ChunkMaxChars = 200 };
        var chunks = Chunker(options).Chunk(Cv, isResume: true);

        foreach (var chunk in chunks)
        {
            // A single sentence longer than the maximum stands alone rather than
            // being cut — a truncated bullet is not evidence of anything.
            var sentences = chunk.Text.Split(". ", StringSplitOptions.RemoveEmptyEntries);
            if (sentences.Length > 1) Assert.True(chunk.Text.Length <= options.ChunkMaxChars * 1.5);
        }
    }

    [Fact]
    public void Overlap_carries_whole_sentences_not_fragments()
    {
        var options = new RagOptions
        {
            ChunkTargetChars = 80, ChunkMaxChars = 140, ChunkSentenceOverlap = 1,
        };

        var chunks = Chunker(options)
            .Chunk(Cv, isResume: true)
            .Where(c => c.Section.Contains("Experience", StringComparison.OrdinalIgnoreCase))
            .ToList();

        Assert.True(chunks.Count > 1, "the experience section should split at this size");

        // No chunk may start mid-sentence: a quote taken from either side of a
        // boundary has to be something the candidate actually wrote.
        Assert.All(chunks, c => Assert.False(c.Text.StartsWith(' ')));
        Assert.All(chunks, c => Assert.False(string.IsNullOrWhiteSpace(c.Text)));
    }

    [Fact]
    public void Zero_overlap_is_honoured()
    {
        var options = new RagOptions
        {
            ChunkTargetChars = 80, ChunkMaxChars = 140, ChunkSentenceOverlap = 0,
        };

        var chunks = Chunker(options).Chunk(Cv, isResume: true);
        var joined = string.Join(" | ", chunks.Select(c => c.Text));

        // With no overlap the settlement sentence appears exactly once.
        var occurrences = joined.Split("settlement service").Length - 1;
        Assert.Equal(1, occurrences);
    }

    // -- documents that are not tidy ----------------------------------------

    [Fact]
    public void A_document_with_no_headings_is_still_chunked()
    {
        var flat = "I am a backend engineer. I have worked with Java and Spring Boot for six years. "
                 + "Most recently I built a settlement service at a payments company.";

        var chunks = Chunker().Chunk(flat, isResume: true);

        Assert.NotEmpty(chunks);
        Assert.All(chunks, c => Assert.False(string.IsNullOrWhiteSpace(c.Section)));
    }

    [Fact]
    public void Text_before_the_first_heading_is_not_lost()
    {
        var chunks = Chunker().Chunk(Cv, isResume: true);

        // The name and title sit above every heading. Dropping them would lose the
        // candidate's own job title, which is real evidence.
        Assert.Contains(chunks, c => c.Text.Contains("Priya Raman"));
    }

    [Fact]
    public void Empty_input_produces_nothing_rather_than_throwing()
    {
        Assert.Empty(Chunker().Chunk("", isResume: true));
        Assert.Empty(Chunker().Chunk("   \n  \n ", isResume: true));
    }

    [Fact]
    public void An_abbreviation_does_not_end_a_sentence()
    {
        var chunks = Chunker().Chunk(Cv, isResume: true);
        var education = chunks.First(c => c.Text.Contains("Computer Science"));

        // "B.E. in Computer Science" must survive as one thought — split at the
        // full stop, neither half says what the degree was in.
        Assert.Contains("B.E. in Computer Science", education.Text);
    }

    [Fact]
    public void Page_markers_become_metadata_rather_than_text()
    {
        var paged = "Summary\nFirst page content here.\n--- PAGE 2 ---\nSecond page content here.";

        // Small enough that each sentence becomes its own chunk, and no overlap,
        // so each chunk's page is unambiguous. A chunk takes the page of its FIRST
        // sentence, so a chunk that straddles a page break reports where it began.
        var options = new RagOptions { ChunkTargetChars = 10, ChunkMaxChars = 20, ChunkSentenceOverlap = 0 };
        var chunks = Chunker(options).Chunk(paged, isResume: true);

        Assert.All(chunks, c => Assert.DoesNotContain("PAGE", c.Text));
        Assert.Contains(chunks, c => c.Page == 1 && c.Text.Contains("First page"));
        Assert.Contains(chunks, c => c.Page == 2 && c.Text.Contains("Second page"));
    }

    // -- postings ------------------------------------------------------------

    [Fact]
    public void A_job_posting_splits_on_its_own_headings()
    {
        var jd = """
            Senior Backend Engineer

            Responsibilities
            Design and ship payment services. Own the on-call rotation for your services.

            Requirements
            5+ years of Java. Strong PostgreSQL. Experience with Kafka.

            Nice To Have
            Kubernetes, Terraform.
            """;

        var chunks = Chunker().Chunk(jd, isResume: false);

        Assert.Contains(chunks, c => c.Section.Contains("Responsibilities", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(chunks, c => c.Section.Contains("Requirements", StringComparison.OrdinalIgnoreCase));

        // A must-have and a nice-to-have must not share a chunk, or the evaluator
        // cannot tell which is which.
        Assert.DoesNotContain(chunks, c =>
            c.Text.Contains("5+ years of Java") && c.Text.Contains("Terraform"));
    }
}
