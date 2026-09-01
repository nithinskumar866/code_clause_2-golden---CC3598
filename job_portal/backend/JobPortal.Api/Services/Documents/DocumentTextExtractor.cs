using System.Text;
using System.Text.RegularExpressions;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;

namespace JobPortal.Api.Services.Documents;

public interface IDocumentTextExtractor
{
    bool IsSupported(string filename);
    string Extract(Stream stream, string filename);
    string Extract(string path);
}

/// <summary>
/// Turns an uploaded PDF, DOCX or plain-text file into text.
///
/// This is the only place in the portal that touches file formats. Everything
/// downstream (extraction, embedding, matching, the chat) sees text and nothing
/// else, which is what keeps a new format from rippling through the system.
/// </summary>
public class DocumentTextExtractor : IDocumentTextExtractor
{
    private static readonly string[] Supported = { ".pdf", ".docx", ".txt", ".md" };

    // Three or more blank lines carry no information and cost tokens in every
    // prompt built from this text.
    private static readonly Regex ExcessBlankLines = new(@"(\r?\n){3,}", RegexOptions.Compiled);
    private static readonly Regex TrailingSpaces = new(@"[ \t]+(\r?\n)", RegexOptions.Compiled);

    private readonly ILogger<DocumentTextExtractor> _logger;

    public DocumentTextExtractor(ILogger<DocumentTextExtractor> logger) => _logger = logger;

    public bool IsSupported(string filename) =>
        Supported.Contains(Path.GetExtension(filename).ToLowerInvariant());

    public string Extract(string path)
    {
        using var stream = File.OpenRead(path);
        return Extract(stream, path);
    }

    public string Extract(Stream stream, string filename)
    {
        var extension = Path.GetExtension(filename).ToLowerInvariant();

        // PdfPig and OpenXml both need to seek. An ASP.NET form stream generally
        // can, but a buffered copy makes that independent of the caller.
        var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        buffer.Position = 0;

        var text = extension switch
        {
            ".pdf" => ExtractPdf(buffer, filename),
            ".docx" => ExtractDocx(buffer, filename),
            ".txt" or ".md" => new StreamReader(buffer, Encoding.UTF8, detectEncodingFromByteOrderMarks: true).ReadToEnd(),
            _ => throw new NotSupportedException(
                $"'{extension}' is not a supported document type. Upload a PDF, DOCX, TXT or MD file."),
        };

        return Clean(text);
    }

    private string ExtractPdf(Stream stream, string filename)
    {
        try
        {
            using var document = PdfDocument.Open(stream);
            var builder = new StringBuilder();

            foreach (var page in document.GetPages())
            {
                // ContentOrderTextExtractor follows the PDF's own content stream
                // order, which keeps two-column resumes from interleaving the way
                // a naive left-to-right sweep over word positions does.
                var pageText = ContentOrderTextExtractor.GetText(page);
                if (!string.IsNullOrWhiteSpace(pageText))
                {
                    builder.AppendLine(pageText);
                    builder.AppendLine();
                }
            }

            var text = builder.ToString();
            if (string.IsNullOrWhiteSpace(text))
            {
                // Almost always a scanned document: real pages, no text layer.
                // Saying so beats returning an empty string that looks like an
                // empty resume to everything downstream.
                throw new InvalidOperationException(
                    $"No text could be read from '{filename}'. If it is a scanned document, " +
                    "it needs OCR before it can be processed.");
            }
            return text;
        }
        catch (Exception ex) when (ex is not InvalidOperationException && ex is not NotSupportedException)
        {
            _logger.LogError(ex, "Failed to read PDF {Filename}", filename);
            throw new InvalidOperationException($"'{filename}' could not be read as a PDF.", ex);
        }
    }

    private string ExtractDocx(Stream stream, string filename)
    {
        try
        {
            using var document = WordprocessingDocument.Open(stream, isEditable: false);
            var body = document.MainDocumentPart?.Document?.Body;
            if (body is null) return "";

            var builder = new StringBuilder();

            foreach (var element in body.Elements())
            {
                switch (element)
                {
                    case Paragraph paragraph:
                        builder.AppendLine(ParagraphText(paragraph));
                        break;

                    // Job descriptions and resumes both use tables for layout, and
                    // dropping them would lose whole skills sections. Cells are
                    // joined with a tab so a row stays one line.
                    case Table table:
                        foreach (var row in table.Elements<TableRow>())
                        {
                            var cells = row.Elements<TableCell>()
                                .Select(c => string.Join(" ", c.Descendants<Paragraph>().Select(ParagraphText)).Trim())
                                .Where(c => c.Length > 0);
                            builder.AppendLine(string.Join("\t", cells));
                        }
                        builder.AppendLine();
                        break;
                }
            }

            return builder.ToString();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to read DOCX {Filename}", filename);
            throw new InvalidOperationException($"'{filename}' could not be read as a DOCX file.", ex);
        }
    }

    private static string ParagraphText(Paragraph paragraph)
    {
        var builder = new StringBuilder();
        foreach (var run in paragraph.Descendants<Run>())
        {
            foreach (var child in run.ChildElements)
            {
                switch (child)
                {
                    case Text text: builder.Append(text.Text); break;
                    case TabChar: builder.Append('\t'); break;
                    // A soft line break inside a run is a real line break to a
                    // reader, and joining across it merges two bullet points.
                    case Break: builder.Append('\n'); break;
                }
            }
        }
        return builder.ToString();
    }

    private static string Clean(string text)
    {
        text = text.Replace("\r\n", "\n").Replace('\r', '\n');
        text = TrailingSpaces.Replace(text, "$1");
        text = ExcessBlankLines.Replace(text, "\n\n");
        return text.Trim();
    }
}
