using JobPortal.Api.Options;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace JobPortal.Api.Tests;

/// <summary>
/// The shipped appsettings.json must parse and must actually switch the models on.
///
/// Worth a test because both failure modes are silent in different ways: a config
/// file the provider cannot read takes the whole API down at startup, and a config
/// that parses but leaves BaseUrl or ChatModel blank leaves <c>ChatAvailable</c>
/// false — so every LLM feature quietly does nothing and the portal looks merely
/// dull rather than misconfigured.
/// </summary>
public class AppSettingsTests
{
    private static OllamaOptions Load()
    {
        // The API's own file, copied to the output by the csproj. Reading it from
        // the output rather than by walking the source tree keeps this working
        // wherever the build puts its artifacts.
        var path = Path.Combine(AppContext.BaseDirectory, "api.appsettings.json");
        Assert.True(File.Exists(path), $"appsettings.json was not copied to {path}");

        var configuration = new ConfigurationBuilder().AddJsonFile(path).Build();
        var options = new OllamaOptions();
        configuration.GetSection("Ollama").Bind(options);
        return options;
    }

    [Fact]
    public void The_config_file_parses()
    {
        // Also proves the JSON configuration provider tolerates the // comments in it.
        var options = Load();
        Assert.NotNull(options);
    }

    [Fact]
    public void Chat_is_switched_on()
    {
        var options = Load();
        Assert.True(options.Enabled,
            "BaseUrl and ChatModel must both be set or every LLM feature silently no-ops.");
    }

    [Fact]
    public void Embeddings_are_switched_on()
    {
        Assert.True(Load().EmbeddingsEnabled);
    }

    [Fact]
    public void The_base_url_is_not_the_openai_compatible_form()
    {
        // This client posts to native api/chat. A /v1 base makes every call 404 —
        // and the Python platform, pointed at the SAME pod, does need /v1, so the
        // two URLs sit side by side in the repo inviting exactly this mistake.
        var url = Load().BaseUrl.TrimEnd('/');
        Assert.False(url.EndsWith("/v1"), $"Native Ollama base must not end in /v1: {url}");
    }
}
