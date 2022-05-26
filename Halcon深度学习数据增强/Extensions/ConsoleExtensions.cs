using System;

namespace Halcon深度学习数据增强.Extensions;

internal static class ConsoleExtensions
{
    public static void ConsoleSplitLine(char splitLineChar = '_',
        ConsoleColor foregroundColor = ConsoleColor.Gray,
        ConsoleColor backgroundColor = ConsoleColor.Black)
    {
        var width = Console.WindowWidth;

        new string(splitLineChar, width - 1).WriteLine(foregroundColor,
            backgroundColor);
    }

    public static void WriteLine<T>(this T t,
        ConsoleColor foregroundColor = ConsoleColor.Gray,
        ConsoleColor backgroundColor = ConsoleColor.Black)
    {
        lock (Console.Out)
        {
            var backgroundBuff = Console.BackgroundColor;
            var foregroundBuff = Console.ForegroundColor;
            Console.BackgroundColor = backgroundColor;
            Console.ForegroundColor = foregroundColor;
            Console.WriteLine(t);
            Console.BackgroundColor = backgroundBuff;
            Console.ForegroundColor = foregroundBuff;
        }
    }
}