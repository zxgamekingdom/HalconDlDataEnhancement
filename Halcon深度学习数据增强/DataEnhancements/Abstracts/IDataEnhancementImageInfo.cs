using HalconDotNet;

namespace Halcon深度学习数据增强.DataEnhancements.Abstracts;

public interface IDataEnhancementImageInfo
{
    public string FileName { get; }

    public long Id { get; }

    public HImage Image { get; }
}
