using System;
using System.Collections.Generic;
using System.Linq;
using HalconDotNet;
using Halcon深度学习数据增强.Dicts.Extensions;

namespace Halcon深度学习数据增强.Dicts;

public class HalconOrientedObjectDetectionDict
{

    public List<long>? Ids { get; set; }

    public List<string>? Names { get; set; }

    public string? ImageDir { get; set; }

    public List<Sample>? Samples { get; set; }

    public static HalconOrientedObjectDetectionDict FromHDict(HDict dict)
    {
        var buff = new HalconOrientedObjectDetectionDict
        {
            Ids = dict.FromKey("class_ids", x => x.LArr.ToList()),
            Names = dict.FromKey("class_names", x => x.SArr.ToList()),
            ImageDir = dict.FromKey("image_dir", x => x.S)
        };

        var handles = dict.FromKey("samples", x => x.HArr);

        if (handles == null) return buff;

        var samplesLength = handles.Length;
        var list = new List<Sample>(samplesLength + 10);

        for (var i = 0; i < samplesLength; i++)
        {
            var d = new HDict(handles[i]);

            list.Add(new Sample
            {
                Id = d.FromKey("image_id", x => x.L),
                FileName = d.FromKey("image_file_name", x => x.S),
                BboxLabelId = d.FromKey("bbox_label_id", x => x.LArr.ToList()),
                BboxRow = d.FromKey("bbox_row", x => x.DArr.ToList()),
                BboxCol = d.FromKey("bbox_col", x => x.DArr.ToList()),
                BboxLength1 = d.FromKey("bbox_length1", x => x.DArr.ToList()),
                BboxLength2 = d.FromKey("bbox_length2", x => x.DArr.ToList()),
                BboxPhi = d.FromKey("bbox_phi", x => x.DArr.ToList())
            });
        }

        buff.Samples = list;

        return buff;
    }

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

        public List<long>? BboxLabelId { get; set; }

        public List<double>? BboxRow { get; set; }

        public List<double>? BboxCol { get; set; }

        public List<double>? BboxLength1 { get; set; }

        public List<double>? BboxLength2 { get; set; }

        public List<double>? BboxPhi { get; set; }

        public HDict ToHDict()
        {
            var dict = new HDict();
            dict.SetDictTuple("image_id", new HTuple(Id));
            dict.SetDictTuple("image_file_name", new HTuple(FileName));
            dict.SetDictTuple("bbox_label_id", new HTuple(BboxLabelId?.ToArray()));
            dict.SetDictTuple("bbox_row", new HTuple(BboxRow?.ToArray()));
            dict.SetDictTuple("bbox_col", new HTuple(BboxCol?.ToArray()));
            dict.SetDictTuple("bbox_length1", new HTuple(BboxLength1?.ToArray()));
            dict.SetDictTuple("bbox_length2", new HTuple(BboxLength2?.ToArray()));
            dict.SetDictTuple("bbox_phi", new HTuple(BboxPhi?.ToArray()));

            return dict;
        }

        public IEnumerable<string> Errors()
        {
            var errors = new List<string>();
            if (Id == null) errors.Add($"{nameof(Id)}为空");
            if (FileName.IsNullOrWhiteSpace()) errors.Add($"{nameof(FileName)}为空");
            if (BboxLabelId == null) errors.Add($"{nameof(BboxLabelId)}为空");
            if (BboxRow == null) errors.Add($"{nameof(BboxRow)}为空");
            if (BboxCol == null) errors.Add($"{nameof(BboxCol)}为空");
            if (BboxLength1 == null) errors.Add($"{nameof(BboxLength1)}为空");
            if (BboxLength2 == null) errors.Add($"{nameof(BboxLength2)}为空");
            if (BboxPhi == null) errors.Add($"{nameof(BboxPhi)}为空");

            // BboxLabelId BboxRow BboxCol BboxLength1  BboxLength2 BboxPhi 长度不一致
            if (BboxPhi != null &&
                BboxLength2 != null &&
                BboxLength1 != null &&
                BboxCol != null &&
                BboxRow != null &&
                BboxLabelId != null &&
                (BboxLabelId.Count != BboxRow.Count ||
                    BboxLabelId.Count != BboxCol.Count ||
                    BboxLabelId.Count != BboxLength1.Count ||
                    BboxLabelId.Count != BboxLength2.Count ||
                    BboxLabelId.Count != BboxPhi.Count))
                errors.Add(
                    $"{nameof(BboxLabelId)}、{nameof(BboxRow)}、{nameof(BboxCol)}、{nameof(BboxLength1)}、{nameof(BboxLength2)}、{nameof(BboxPhi)}长度不一致");

            return errors;
        }

    }

}

public class HalconSemanticSegmentationDict
{

}

public class HalconInstanceSegmentationDict
{

}
