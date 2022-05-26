using System;
using System.Collections.Generic;
using Halcon深度学习数据增强.Abstracts;

namespace Halcon深度学习数据增强.Dicts.Tools;

public struct ImageInfoComparer : IEqualityComparer<IImageInfo>
{
    public bool Equals(IImageInfo x, IImageInfo y)
    {
        if (ReferenceEquals(x, y)) return true;
        if (ReferenceEquals(x, null)) return false;
        if (ReferenceEquals(y, null)) return false;
        if (x.GetType() != y.GetType()) return false;

        return string.Equals(x.FileName,
                y.FileName,
                StringComparison.CurrentCultureIgnoreCase) &&
            x.Id == y.Id;
    }

    public int GetHashCode(IImageInfo obj)
    {
        unchecked
        {
            return ((obj.FileName != null
                        ? StringComparer.CurrentCultureIgnoreCase.GetHashCode(
                            obj.FileName)
                        : 0) *
                    397) ^
                obj.Id.GetHashCode();
        }
    }
}
