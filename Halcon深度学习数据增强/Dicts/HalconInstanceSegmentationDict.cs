using System.Collections.Generic;
using System.Linq;
using HalconDotNet;
using Halcon深度学习数据增强.Abstracts;
using Halcon深度学习数据增强.Dicts.Abstracts;
using Halcon深度学习数据增强.Dicts.Extensions;
using Halcon深度学习数据增强.Dicts.Tools;

namespace Halcon深度学习数据增强.Dicts;

public class
    HalconInstanceSegmentationDict : IHalconDict<HalconInstanceSegmentationDict.Sample>
{
    public List<long>? Ids { get; set; }

    public string? ImageDir { get; set; }

    public List<string>? Names { get; set; }

    public List<Sample>? Samples { get; set; }

    public void FromHDict(HDict dict)
    {
        Ids = dict.FromKeyTuple("class_ids", x => x.LArr.ToList());
        Names = dict.FromKeyTuple("class_names", x => x.SArr.ToList());
        ImageDir = dict.FromKeyTuple("image_dir", x => x.S);
        var samples = dict.FromKeyTuple("samples", x => x.HArr);

        if (samples == null) return;

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
    }

    public IEnumerable<string> Errors()
    {
        var errorList = new List<string>();
        if (Ids == null) errorList.Add($"{nameof(Ids)}为空");
        if (ImageDir.IsNullOrWhiteSpace()) errorList.Add("图片路径为空");
        if (Names == null) errorList.Add($"{nameof(Names)}为空");
        if (Samples == null) errorList.Add($"{nameof(Samples)}为空");

        if (Names != null && Ids != null && Ids.Count != Names.Count)
            errorList.Add($"{nameof(Ids)}与{nameof(Names)}数量不一致");

        Ids?.检查重复(errorList, l => $"有重复的Id:{l}");
        Names?.检查重复(errorList, l => $"有重复的Name:{l}");
        Samples?.检查重复(errorList, l => $"有重复的Sample:{l}", new ImageInfoComparer());

        if (Samples != null)
            foreach (var sample in Samples)
                errorList.AddRange(sample.Errors());

        return errorList;
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

    public class Sample : IImageInfo, IHalconErrors
    {
        public List<double>? BboxCol1 { get; set; }

        public List<double>? BboxCol2 { get; set; }

        public List<long>? BboxLabelId { get; set; }

        public List<double>? BboxRow1 { get; set; }

        public List<double>? BboxRow2 { get; set; }

        public List<HRegion>? Mask { get; set; }

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

        public string? FileName { get; set; }

        public long? Id { get; set; }

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
    }
}
