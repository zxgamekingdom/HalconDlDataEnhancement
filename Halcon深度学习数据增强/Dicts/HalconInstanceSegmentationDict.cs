﻿using System;
using System.Collections.Generic;
using System.Linq;
using HalconDotNet;
using Halcon深度学习数据增强.Dicts.Extensions;

namespace Halcon深度学习数据增强.Dicts;

public class HalconInstanceSegmentationDict
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

    public static HalconInstanceSegmentationDict FromHDict(HDict dict)
    {
        var buff = new HalconInstanceSegmentationDict
        {
            Ids = dict.FromKeyTuple("class_ids", x => x.LArr.ToList()),
            Names = dict.FromKeyTuple("class_names", x => x.SArr.ToList()),
            ImageDir = dict.FromKeyTuple("image_dir", x => x.S)
        };

        var samples = dict.FromKeyTuple("samples", x => x.HArr);

        if (samples == null) return buff;

        var samplesLength = samples.Length;
        var list = new List<Sample>(samplesLength + 10);

        for (var i = 0; i < samplesLength; i++)
        {
            var d = new HDict(samples[i]);

            list.Add(new Sample
            {
                Id = d.FromKeyTuple("image_id", x => x.L),
                FileName = d.FromKeyTuple("image_file_name", x => x.S),
                Mask = HRegionListFromHDict(d, "mask"),
                BboxLabelId = d.FromKeyTuple("bbox_label_id", x => x.LArr.ToList()),
                BboxRow1 = d.FromKeyTuple("bbox_row1", x => x.DArr.ToList()),
                BboxCol1 = d.FromKeyTuple("bbox_col1", x => x.DArr.ToList()),
                BboxRow2 = d.FromKeyTuple("bbox_row2", x => x.DArr.ToList()),
                BboxCol2 = d.FromKeyTuple("bbox_col2", x => x.DArr.ToList())
            });
        }

        buff.Samples = list;

        return buff;
    }

    private static List<HRegion>? HRegionListFromHDict(HDict dict, string key)
    {
        var hObject = dict.GetDictObject(key);
        var countObj = hObject.CountObj();

        if (countObj == 0) return null;

        var regions = new List<HRegion>(countObj);

        for (var i = 1; i <= countObj; i++)
        {
            var o = hObject[i];
            var hRegion = new HRegion(o);
            regions.Add(hRegion);
        }

        return regions;
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

        public List<HRegion>? Mask { get; set; }

        public List<long>? BboxLabelId { get; set; }

        public List<double>? BboxRow1 { get; set; }

        public List<double>? BboxCol1 { get; set; }

        public List<double>? BboxRow2 { get; set; }

        public List<double>? BboxCol2 { get; set; }

        public HDict ToHDict()
        {
            var dict = new HDict();
            dict.SetDictTuple("image_id", new HTuple(Id));
            dict.SetDictTuple("image_file_name", new HTuple(FileName));
            HRegion? m = default;

            if (Mask != null)
            {
                m ??= new HRegion();
                m.GenEmptyObj();
                foreach (var hRegion in Mask) m = m.ConcatObj(hRegion);
            }

            if (m != null) dict.SetDictObject(m, "mask");
            dict.SetDictTuple("bbox_label_id", new HTuple(BboxLabelId?.ToArray()));
            dict.SetDictTuple("bbox_row1", new HTuple(BboxRow1?.ToArray()));
            dict.SetDictTuple("bbox_col1", new HTuple(BboxCol1?.ToArray()));
            dict.SetDictTuple("bbox_row2", new HTuple(BboxRow2?.ToArray()));
            dict.SetDictTuple("bbox_col2", new HTuple(BboxCol2?.ToArray()));

            return dict;
        }

        public IEnumerable<string> Errors()
        {
            var errors = new List<string>();
            if (Id == null) errors.Add($"{nameof(Id)}为空");
            if (FileName.IsNullOrWhiteSpace()) errors.Add($"{nameof(FileName)}为空");
            if (Mask == null) errors.Add($"{nameof(Mask)}为空");
            if (BboxLabelId == null) errors.Add($"{nameof(BboxLabelId)}为空");
            if (BboxRow1 == null) errors.Add($"{nameof(BboxRow1)}为空");
            if (BboxCol1 == null) errors.Add($"{nameof(BboxCol1)}为空");
            if (BboxRow2 == null) errors.Add($"{nameof(BboxRow2)}为空");
            if (BboxCol2 == null) errors.Add($"{nameof(BboxCol2)}为空");

            if (BboxLabelId != null && Mask != null && Mask.Count != BboxLabelId.Count)
                errors.Add($"{nameof(Mask)}与{nameof(BboxLabelId)}数量不一致");

            if (BboxRow1 != null && Mask != null && Mask.Count != BboxRow1.Count)
                errors.Add($"{nameof(Mask)}与{nameof(BboxRow1)}数量不一致");

            if (BboxCol1 != null && Mask != null && Mask.Count != BboxCol1.Count)
                errors.Add($"{nameof(Mask)}与{nameof(BboxCol1)}数量不一致");

            if (BboxRow2 != null && Mask != null && Mask.Count != BboxRow2.Count)
                errors.Add($"{nameof(Mask)}与{nameof(BboxRow2)}数量不一致");

            if (BboxCol2 != null && Mask != null && Mask.Count != BboxCol2.Count)
                errors.Add($"{nameof(Mask)}与{nameof(BboxCol2)}数量不一致");

            return errors;
        }

    }

}
