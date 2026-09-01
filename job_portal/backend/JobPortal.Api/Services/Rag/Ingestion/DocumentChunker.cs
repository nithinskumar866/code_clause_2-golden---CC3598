using System.Text;
using System.Text.RegularExpressions;
using JobPortal.Api.Services.Documents;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Rag.Ingestion;

/// <summary>One piece of a document, with where it came from.</summary>
public record DocumentChunk(string Text, string Section, int Ordinal, int Page);

/// <summary>
/// Splits a document into retrievable pieces.
///
/// Two levels, and the order matters:
///
///   1. SECTIONS. A chunk never spans a section boundary. This is the whole point
///      of chunking a résumé rather than a novel: "React" under *Skills* is a
///      claim and "React" under *Experience* is proof, and a chunk straddling the
///      two makes that distinction unrecoverable downstream.
///
///   2. SENTENCES, packed. Inside a section, whole sentences accumulate until the
///      chunk is big enough and the next sentence would make it too big.
///
/// Overlap is measured in SENTENCES, not characters. A character window cuts
/// mid-sentence, and a quote pulled from either side of that cut is not something
/// the candidate actually wrote — which matters because these chunks are shown to
/// people as evidence.
///
/// Sizing is ported from the Python platform, where it was tuned against real
/// résumés: 350 target, 600 hard maximum, one sentence of overlap.
/// </summary>
public class DocumentChunker
{
    private readonly RagOptions _options;

    public DocumentChunker(IOptions<RagOptions> options) => _options = options.Value;

    /// <summary>
    /// Section headings recognised in a résumé.
    ///
    /// Superset of the ones <see cref="Matching.ResumeEvidenceService"/> uses, so
    /// a chunk's section label lines up with the evidence tier the rest of the
    /// platform already assigns to that heading.
    /// </summary>
    private static readonly string[] ResumeHeadings =
    {
        "Experience", "Work Experience", "Professional Experience", "Employment History",
        "Employment", "Career History", "Work History", "Internship", "Internships",
        "Education", "Academic Background", "Qualifications",
        "Certifications", "Certificates", "Licenses",
        "Projects", "Personal Projects", "Academic Projects", "Key Projects",
        "Skills", "Technical Skills", "Key Skills", "Core Competencies", "Competencies",
        "Summary", "Profile", "Objective", "About", "About Me",
        "Achievements", "Awards", "Publications", "Interests",
    };

    /// <summary>Headings a job posting tends to use.</summary>
    private static readonly string[] JobHeadings =
    {
        "Responsibilities", "Key Responsibilities", "What You'll Do", "The Role",
        "Requirements", "Qualifications", "Required Skills", "Must Have",
        "Preferred", "Preferred Qualifications", "Nice To Have", "Bonus",
        "About", "About Us", "About The Role", "Benefits", "Perks",
        "Experience", "Education", "Skills",
    };

    /// <summary>Text before any recognised heading, and the label it gets.</summary>
    private const string Preamble = "Summary";

    /// <summary>
    /// A sentence boundary: terminal punctuation, whitespace, then something that
    /// starts a sentence. Kept conservative — over-splitting scatters one thought
    /// across two chunks, which is worse than a slightly long chunk.
    /// </summary>
    private static readonly Regex SentenceBreak =
        new(@"(?<=[.!?])\s+(?=[A-Z0-9""'•\-])", RegexOptions.Compiled);

    /// <summary>The page marker the document loader emits.</summary>
    private static readonly Regex PageMarker =
        new(@"^---\s*PAGE\s+(\d+)\s*---$", RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>
    /// Abbreviations that end in a full stop without ending a sentence. Without
    /// this, "B.S. in Computer Science" becomes two chunks and neither says what
    /// the degree was in.
    /// </summary>
    private static readonly HashSet<string> Abbreviations = new(StringComparer.OrdinalIgnoreCase)
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
        "inc", "ltd", "co", "corp", "llc", "plc",
        "e.g", "i.e", "etc", "vs", "approx", "dept", "univ",
        "b.s", "m.s", "b.a", "m.a", "b.e", "b.tech", "m.tech", "ph.d", "mba",
    };

    public IReadOnlyList<DocumentChunk> Chunk(string text, bool isResume)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<DocumentChunk>();

        var headings = isResume ? ResumeHeadings : JobHeadings;
        var chunks = new List<DocumentChunk>();
        var ordinal = 0;

        foreach (var (section, body) in Sections(text, headings))
        {
            // The packer works one section at a time and has no reason to know
            // which; the label and the document-wide ordinal are attached here,
            // where both are known.
            foreach (var chunk in PackSection(body))
            {
                chunks.Add(chunk with { Section = section, Ordinal = ordinal++ });
            }
        }

        return chunks;
    }

    /// <summary>
    /// The document's sections, each with its body.
    ///
    /// <see cref="TextStructure.SplitSections"/> already keeps everything before
    /// the first heading — under an empty key, because that is the section name it
    /// started with. That text is usually the candidate's name, title and summary,
    /// which is real evidence, so it is relabelled rather than dropped or left
    /// blank. Emitting it separately as well is what produced every chunk twice.
    /// </summary>
    private static IEnumerable<(string Section, string Body)> Sections(
        string text, IReadOnlyCollection<string> headings)
    {
        var known = TextStructure.SplitSections(text, headings);

        // No recognised heading anywhere — a flat document. One section rather
        // than none: an unstructured CV must still be searchable.
        if (known.Count == 0)
        {
            yield return (Preamble, text);
            yield break;
        }

        foreach (var (section, body) in known)
        {
            if (string.IsNullOrWhiteSpace(body)) continue;
            yield return (string.IsNullOrWhiteSpace(section) ? Preamble : section, body);
        }
    }

    /// <summary>
    /// A run of lines that belong to one thought, and the page it started on.
    ///
    /// PDF extraction wraps a sentence across several lines, so splitting
    /// sentences line by line produces fragments: a chunk beginning
    /// "framework. ● Collaborated with…" is the tail of a sentence whose head is
    /// in the previous chunk. Those get quoted to candidates as evidence, so the
    /// wrap has to be undone before any sentence splitting happens.
    ///
    /// A paragraph ends at a blank line or at the start of a new bullet — bullets
    /// are separate thoughts even when the one above them has no full stop.
    /// </summary>
    private record Paragraph(string Text, int Page);

    private static readonly Regex BulletStart =
        new(@"^\s*[•●▪◦‣·*\-–—]\s+", RegexOptions.Compiled);

    private static IEnumerable<Paragraph> Paragraphs(string body)
    {
        var page = 1;
        var buffer = new StringBuilder();
        var startPage = 1;

        foreach (var raw in body.Split('\n'))
        {
            var pageMatch = PageMarker.Match(raw.Trim());
            if (pageMatch.Success)
            {
                // A page break closes the paragraph. It costs the ability to
                // rejoin a sentence that wraps across pages, which is rare;
                // keeping it open costs correct page attribution for everything
                // after the break, which is not.
                if (buffer.Length > 0)
                {
                    yield return new Paragraph(buffer.ToString(), startPage);
                    buffer.Clear();
                }

                page = int.Parse(pageMatch.Groups[1].Value);
                continue;
            }

            var clean = TextStructure.Collapse(raw);

            // Blank line, or a new bullet: whatever was accumulating is complete.
            if (clean.Length == 0 || BulletStart.IsMatch(raw))
            {
                if (buffer.Length > 0)
                {
                    yield return new Paragraph(buffer.ToString(), startPage);
                    buffer.Clear();
                }
                if (clean.Length == 0) continue;
            }

            if (buffer.Length == 0) startPage = page;
            else buffer.Append(' ');

            buffer.Append(clean);
        }

        if (buffer.Length > 0) yield return new Paragraph(buffer.ToString(), startPage);
    }

    /// <summary>Packs one section's sentences into chunks.</summary>
    private IEnumerable<DocumentChunk> PackSection(string body)
    {
        var buffer = new List<string>();
        var length = 0;
        var chunkPage = 1;

        foreach (var (text, page) in Paragraphs(body))
        {
            foreach (var sentence in Sentences(text))
            {
                if (buffer.Count == 0) chunkPage = page;

                // Flush BEFORE adding, when the buffer is already big enough and
                // this sentence would push it past the hard maximum.
                if (length >= _options.ChunkTargetChars &&
                    length + sentence.Length > _options.ChunkMaxChars)
                {
                    yield return Emit(buffer, chunkPage);
                    buffer = Carry(buffer);
                    length = buffer.Sum(s => s.Length + 1);
                    chunkPage = page;
                }

                buffer.Add(sentence);
                length += sentence.Length + 1;

                // A single sentence longer than the maximum stands alone rather
                // than being cut — a truncated bullet is not evidence of anything.
                if (buffer.Count == 1 && length > _options.ChunkMaxChars)
                {
                    yield return Emit(buffer, chunkPage);
                    buffer = new List<string>();
                    length = 0;
                }
            }
        }

        if (buffer.Count > 0) yield return Emit(buffer, chunkPage);
    }

    private static DocumentChunk Emit(List<string> buffer, int page) =>
        // Ordinal is assigned by the caller, which is the only place that knows the
        // document-wide position.
        new(string.Join(" ", buffer), Section: "", Ordinal: 0, Page: page);

    /// <summary>The trailing sentences carried into the next chunk.</summary>
    private List<string> Carry(List<string> buffer)
    {
        var overlap = Math.Clamp(_options.ChunkSentenceOverlap, 0, buffer.Count);
        return overlap == 0 ? new List<string>() : buffer.TakeLast(overlap).ToList();
    }

    /// <summary>
    /// Sentences within one line.
    ///
    /// A bullet with no terminal punctuation stays whole — résumé bullets are the
    /// atomic unit worth keeping, and splitting one produces two fragments that
    /// each mean less than the whole.
    /// </summary>
    private static IEnumerable<string> Sentences(string line)
    {
        var parts = SentenceBreak.Split(line);
        var merged = new List<string>();

        foreach (var part in parts)
        {
            var piece = part.Trim();
            if (piece.Length == 0) continue;

            // Re-join a split that landed after an abbreviation or a single
            // initial, both of which end in a full stop without ending a sentence.
            if (merged.Count > 0 && EndsWithAbbreviation(merged[^1]))
            {
                merged[^1] = $"{merged[^1]} {piece}";
                continue;
            }

            merged.Add(piece);
        }

        return merged;
    }

    private static bool EndsWithAbbreviation(string text)
    {
        var lastSpace = text.LastIndexOf(' ');
        var lastWord = (lastSpace < 0 ? text : text[(lastSpace + 1)..]).TrimEnd('.');

        if (lastWord.Length == 0) return false;
        if (lastWord.Length == 1 && char.IsUpper(lastWord[0])) return true;  // "B." in "B. Tech"
        return Abbreviations.Contains(lastWord);
    }
}
