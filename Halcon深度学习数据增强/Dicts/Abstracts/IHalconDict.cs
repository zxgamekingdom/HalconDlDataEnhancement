using System.Collections.Generic;
using HalconDotNet;

namespace Halcon深度学习数据增强.Dicts.Abstracts;

public interface IHalconDict<TSample> : IHalconErrors
{
    public List<long>? Ids { get; }

    public string? ImageDir { get; }

    public List<string>? Names { get; }

    public List<TSample>? Samples { get; }

    public void FromHDict(HDict dict);

    public HDict ToHDict();
}
