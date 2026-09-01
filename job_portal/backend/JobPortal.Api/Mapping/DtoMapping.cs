using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;

namespace JobPortal.Api.Mapping;

/// <summary>
/// Entity to contract translation, in one place.
///
/// Entities store list-shaped fields as newline-separated text; the API exposes
/// them as arrays. Doing that conversion here means neither the storage decision
/// nor the wire format leaks into the services.
/// </summary>
public static class DtoMapping
{
    public static JobSummaryDto ToSummaryDto(this PortalJob job, bool isIndexed) => new(
        job.Id,
        job.Title,
        job.Company,
        job.Location,
        job.WorkMode,
        job.EmploymentType,
        job.SeniorityLevel,
        job.MinYearsExperience,
        job.MaxYearsExperience,
        job.SalaryMin,
        job.SalaryMax,
        job.SalaryCurrency,
        TextStructure.ReadLines(job.RequiredSkills),
        TextStructure.ReadLines(job.PreferredSkills),
        job.Summary,
        job.ApplyUrl,
        isIndexed,
        job.CreatedAt);

    public static JobDetailDto ToDetailDto(this PortalJob job, bool isIndexed) => new(
        job.ToSummaryDto(isIndexed),
        TextStructure.ReadLines(job.Responsibilities),
        TextStructure.ReadLines(job.Qualifications),
        job.RawText,
        job.ExtractionMode,
        job.SourceJobDescriptionId);

    public static ResumeProfileDto ToDto(this PortalResume resume) => new(
        resume.Id,
        resume.Filename,
        resume.CandidateName,
        resume.Email,
        resume.Phone,
        resume.Location,
        resume.CurrentTitle,
        resume.YearsExperience,
        TextStructure.ReadLines(resume.Skills),
        TextStructure.ReadLines(resume.Titles),
        TextStructure.ReadLines(resume.Education),
        resume.Summary,
        resume.ExtractionMode);

    public static ChatMessageDto ToDto(this PortalChatMessage message) =>
        new(message.Role, message.Content, message.CreatedAt);
}
