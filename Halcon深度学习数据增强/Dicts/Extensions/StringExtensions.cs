namespace Halcon深度学习数据增强.Dicts.Extensions;

internal static class StringExtensions
{

    public static bool IsNullOrWhiteSpace(this string? fileName)
    {
        return string.IsNullOrWhiteSpace(fileName);
    }

}
