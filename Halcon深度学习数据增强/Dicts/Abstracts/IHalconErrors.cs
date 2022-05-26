using System.Collections.Generic;

namespace Halcon深度学习数据增强.Dicts.Abstracts;

public interface IHalconErrors
{
    public IEnumerable<string> Errors();
}
