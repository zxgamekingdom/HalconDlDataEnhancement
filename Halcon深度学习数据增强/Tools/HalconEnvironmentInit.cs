using HalconDotNet;

namespace Halcon深度学习数据增强.Tools;

public static class HalconEnvironmentInit
{
    public static void Init(int width = 10000, int height = 10000)
    {
        var image = new HImage("byte", width, height);
    }
}