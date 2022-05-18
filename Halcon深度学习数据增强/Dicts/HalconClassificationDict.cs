using System;
using System.Collections.Generic;
using System.Linq;
using HalconDotNet;
using Halcon深度学习数据增强.Dicts.Extensions;

namespace Halcon深度学习数据增强.Dicts;

public class HalconClassificationDict
{

    public List<long>? Ids { get; set; }

    public List<string>? Names { get; set; }

    public string? ImageDir { get; set; }

    public List<Sample>? Samples { get; set; }

    public HDict ToHDict()
    {
        var dict = new HDict();
        dict.SetDictTuple("class_ids", new HTuple(Ids!.ToArray()));
        dict.SetDictTuple("class_names", new HTuple(Names!.ToArray()));
        dict.SetDictTuple("image_dir", new HTuple(ImageDir));
        var tuple = new HTuple();

        if (Samples != null)
            foreach (var hDict in Samples.Select(sample => sample.ToHDict()))
                tuple.Append(hDict);

        dict.SetDictTuple("samples", tuple);

        return dict;
    }

    public static HalconClassificationDict FromHDict(HDict dict)
    {
        var buff = new HalconClassificationDict
        {
            Ids = dict.FromKey("class_ids", x => x.LArr.ToList()),
            Names = dict.FromKey("class_names", x => x.SArr.ToList()),
            ImageDir = dict.FromKey("image_dir", x => x.S)
        };

        var samples = dict.FromKey("samples", x => x.HArr);

        if (samples == null) return buff;

        var samplesLength = samples.Length;
        var list = new List<Sample>(samplesLength + 10);

        for (var i = 0; i < samplesLength; i++)
        {
            var d = new HDict(samples[i]);

            list.Add(new Sample
            {
                Id = d.FromKey("image_id", x => x.L),
                FileName = d.FromKey("image_file_name", x => x.S),
                LabelId = d.FromKey("image_label_id", x => x.L)
            });
        }

        buff.Samples = list;

        return buff;
    }

    public IEnumerable<string> Errors()
    {
        var errorList = new List<string>();
        if (Ids == null) errorList.Add($"{nameof(Ids)}为空");
        if (Names == null) errorList.Add($"{nameof(Names)}为空");

        if (Names != null && Ids != null && Ids.Count != Names.Count)
            errorList.Add($"{nameof(Ids)}与{nameof(Names)}数量不一致");

        if (Ids != null) 检查重复(errorList, Ids, l => $"有重复的Id:{l}");
        if (Names != null) 检查重复(errorList, Names, l => $"有重复的Name:{l}");

        if (Samples != null)
            检查重复(errorList, Samples, l => $"有重复的Sample:{l}", new SampleComparer());

        if (Samples != null)
            foreach (var sample in Samples)
                errorList.AddRange(sample.Errors());

        return errorList;
    }

    private static void 检查重复<T>(ICollection<string> errorList,
        IEnumerable<T> values,
        Func<T, string> 当重复时,
        IEqualityComparer<T>? comparer = default)
    {
        var set = new HashSet<T>(comparer);

        foreach (var value in values)
            if (set.Contains(value))
                errorList.Add(当重复时(value));
            else
                set.Add(value);
    }

    public class SampleComparer : IEqualityComparer<Sample>
    {

        public bool Equals(Sample x, Sample y)
        {
            if (ReferenceEquals(x, y)) return true;
            if (ReferenceEquals(x, null)) return false;
            if (ReferenceEquals(y, null)) return false;
            if (x.GetType() != y.GetType()) return false;

            return x.Id == y.Id &&
                string.Equals(x.FileName,
                    y.FileName,
                    StringComparison.CurrentCultureIgnoreCase);
        }

        public int GetHashCode(Sample obj)
        {
            unchecked
            {
                return (obj.Id.GetHashCode() * 397) ^
                    (obj.FileName != null
                        ? StringComparer.CurrentCultureIgnoreCase.GetHashCode(
                            obj.FileName)
                        : 0);
            }
        }

    }

    public class Sample
    {

        public long? Id { get; set; }

        public string? FileName { get; set; }

        public long? LabelId { get; set; }

        public HDict ToHDict()
        {
            var dict = new HDict();
            dict.SetDictTuple("image_id", new HTuple(Id));
            dict.SetDictTuple("image_file_name", new HTuple(FileName));
            dict.SetDictTuple("image_label_id", new HTuple(LabelId));

            return dict;
        }

        public IEnumerable<string> Errors()
        {
            var errors = new List<string>();
            if (Id == null) errors.Add($"{nameof(Id)}为空");
            if (FileName.IsNullOrWhiteSpace()) errors.Add($"{nameof(FileName)}为空");
            if (LabelId == null) errors.Add($"{nameof(LabelId)}为空");

            return errors;
        }

    }

}
