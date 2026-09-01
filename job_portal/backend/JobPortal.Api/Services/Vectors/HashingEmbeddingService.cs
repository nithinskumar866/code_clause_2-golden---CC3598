using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace JobPortal.Api.Services.Vectors;

/// <summary>
/// The deterministic embedder used when no embedding endpoint is configured or
/// reachable.
///
/// BE CLEAR ABOUT WHAT THIS IS. It projects word and character-trigram counts into
/// a fixed-width space with a stable hash. That gives a real, reproducible, offline
/// similarity measure, and trigrams let it see through spelling and suffix
/// variation ("React" ~ "ReactJS", "Postgres" ~ "PostgreSQL"). It does NOT give
/// semantic adjacency: it cannot know that "React" answers "frontend framework",
/// because nothing in the text of those two phrases overlaps. That is exactly the
/// capability the vibe-matching feature exists to provide, and it requires a real
/// embedding model.
///
/// So this is a floor, not a substitute. The portal stays fully functional and
/// fully testable without a network, and every surface reports which embedder
/// produced the ranking so a lexical result is never mistaken for a semantic one.
/// </summary>
public class HashingEmbeddingService
{
    public const string Id = "deterministic-hashing-v1";

    /// <summary>Namespaces trigram features away from real tokens, so the trigram
    /// "sql" and the word "sql" cannot land in the same bucket by construction.
    /// A control character, because no token the tokenizer produces contains one.</summary>
    private const char TrigramPrefix = (char)1;

    private static readonly Regex TokenPattern = new(@"[a-z0-9][a-z0-9+#\.\-]*", RegexOptions.Compiled);

    // English words that appear in nearly every resume and job ad. They carry no
    // discriminating signal, and leaving them in makes every document look alike.
    private static readonly HashSet<string> StopWords = new(StringComparer.Ordinal)
    {
        "a","an","the","and","or","but","if","then","of","to","in","on","at","for","with","by","from",
        "as","is","are","was","were","be","been","being","it","its","this","that","these","those",
        "we","you","our","your","their","they","he","she","his","her","i","me","my",
        "will","shall","can","could","would","should","may","might","must","have","has","had",
        "do","does","did","not","no","yes","all","any","some","more","most","other","such",
        "into","over","under","about","across","per","via","also","than","them","there","here",
        "work","working","role","team","teams","company","job","position","candidate","applicant",
        "experience","experienced","years","year","strong","good","excellent","great","ability",
        "responsibilities","requirements","required","preferred","plus","etc",
    };

    public int Dimension { get; }

    public HashingEmbeddingService(int dimension = 512)
    {
        if (dimension < 64) throw new ArgumentOutOfRangeException(nameof(dimension));
        Dimension = dimension;
    }

    public IReadOnlyList<float[]> EmbedBatch(IReadOnlyList<string> texts) =>
        texts.Select(Embed).ToList();

    public float[] Embed(string text)
    {
        var vector = new float[Dimension];
        if (string.IsNullOrWhiteSpace(text)) return vector;

        var tokens = Tokenize(text);
        if (tokens.Count == 0) return vector;

        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var token in tokens)
        {
            counts[token] = counts.GetValueOrDefault(token) + 1;
            foreach (var gram in Trigrams(token))
            {
                var key = TrigramPrefix + gram;
                counts[key] = counts.GetValueOrDefault(key) + 1;
            }
        }

        foreach (var (term, count) in counts)
        {
            // Term frequency, sublinearly damped. A job ad that says "Python" nine
            // times is about Python, but not nine times more so than one saying it
            // once; undamped, repetition alone would dominate the direction.
            //
            // Trigrams are a fuzzy-match aid rather than evidence in their own
            // right, and one word emits several of them, so they are discounted.
            // At full weight they would drown the words that produced them.
            var isTrigram = term[0] == TrigramPrefix;
            var weight = (1.0 + Math.Log(count)) * (isTrigram ? 0.25 : 1.0);

            var (index, sign) = Project(term);
            vector[index] += (float)(sign * weight);
        }

        return VectorMath.Normalize(vector);
    }

    public static List<string> Tokenize(string text)
    {
        var tokens = new List<string>();
        foreach (Match match in TokenPattern.Matches(text.ToLowerInvariant()))
        {
            var token = match.Value.Trim('.', '-');
            if (token.Length < 2 || token.Length > 40) continue;
            if (StopWords.Contains(token)) continue;
            tokens.Add(token);
        }
        return tokens;
    }

    private static IEnumerable<string> Trigrams(string token)
    {
        if (token.Length < 4) yield break;
        var padded = "^" + token + "$";
        for (var i = 0; i + 3 <= padded.Length; i++) yield return padded.Substring(i, 3);
    }

    /// <summary>
    /// Maps a term to a bucket and a sign. SHA-256 rather than string.GetHashCode
    /// because vectors are persisted: .NET randomises string hashing per process,
    /// so GetHashCode would make yesterday's stored vectors incomparable with
    /// today's queries. The sign bit is the standard signed-hashing trick: it
    /// makes collisions cancel on average instead of always accumulating.
    /// </summary>
    private (int Index, int Sign) Project(string term)
    {
        Span<byte> digest = stackalloc byte[32];
        SHA256.HashData(Encoding.UTF8.GetBytes(term), digest);

        var raw = BitConverter.ToUInt32(digest[..4]);
        var index = (int)(raw % (uint)Dimension);
        var sign = (digest[4] & 1) == 0 ? 1 : -1;
        return (index, sign);
    }
}
