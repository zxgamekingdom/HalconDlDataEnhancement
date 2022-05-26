using System.Collections.Generic;
using System.Linq;
using HalconDotNet;
using Halcon深度学习数据增强.Abstracts;
using Halcon深度学习数据增强.Dicts.Abstracts;
using Halcon深度学习数据增强.Dicts.Extensions;
using Halcon深度学习数据增强.Dicts.Tools;

namespace Halcon深度学习数据增强.Dicts;

public class
    HalconSemanticSegmentationDict : IHalconDict<HalconSemanticSegmentationDict.Sample>
{

    public List<HDict?>? ClassCustomData { get; set; }

    public string? SegmentationDir { get; set; }

    public List<long>? Ids { get; set; }

    public string? ImageDir { get; set; }

    public List<string>? Names { get; set; }

    public List<Sample>? Samples { get; set; }

    public void FromHDict(HDict dict)
    {
        Ids = dict.FromKeyTuple("class_ids", x => x.LArr.ToList());
        Names = dict.FromKeyTuple("class_names", x => x.SArr.ToList());
        ImageDir = dict.FromKeyTuple("image_dir", x => x.S);
        SegmentationDir = dict.FromKeyTuple("segmentation_dir", x => x.S);

        ClassCustomData = dict.FromKeyTuple("class_custom_data",
            x => x.HArr.Select(handle =>
                {
                    if (handle.IsInitialized() is false) return null;

                    return new HDict(handle);
                })
                .ToList());

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
                SegmentationFileName =
                    d.FromKeyTuple("segmentation_file_name", x => x.S)
            });
        }

        Samples = list;
    }

    public IEnumerable<string> Errors()
    {
        var errorList = new List<string>();
        if (Ids == null) errorList.Add($"{nameof(Ids)}为空");
        if (ImageDir.IsNullOrWhiteSpace()) errorList.Add("图片路径为空");
        if (SegmentationDir.IsNullOrWhiteSpace()) errorList.Add("分割图片路径为空");
        if (Names == null) errorList.Add($"{nameof(Names)}为空");
        if (Samples == null) errorList.Add($"{nameof(Samples)}为空");
        if (ClassCustomData == null) errorList.Add($"{nameof(ClassCustomData)}为空");

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
        var customDataHTuple = new HTuple();

        foreach (var item in ClassCustomData!)
            if (item != null)
                customDataHTuple.Append(item);

        dict.SetDictTuple("class_custom_data", customDataHTuple);
        dict.SetDictTuple("segmentation_dir", new HTuple(SegmentationDir));
        var tuple = new HTuple();

        if (Samples != null)
            foreach (var hDict in Samples.Select(sample => sample.ToHDict()))
                tuple.Append(hDict);

        dict.SetDictTuple("samples", tuple);

        return dict;
    }

    public static class ClassCustomDataConst
    {

        public const string IsBgClass = "is_bg_class";

    }

    public class Sample : IImageInfo, IHalconErrors
    {

        public string? SegmentationFileName { get; set; }

        public IEnumerable<string> Errors()
        {
            var errors = new List<string>();
            if (Id == null) errors.Add($"{nameof(Id)}为空");
            if (FileName.IsNullOrWhiteSpace()) errors.Add($"{nameof(FileName)}为空");

            if (SegmentationFileName.IsNullOrWhiteSpace())
                errors.Add($"{nameof(SegmentationFileName)}为空");

            return errors;
        }

        public string? FileName { get; set; }

        public long? Id { get; set; }

        public HDict ToHDict()
        {
            var dict = new HDict();
            dict.SetDictTuple("image_id", new HTuple(Id));
            dict.SetDictTuple("image_file_name", new HTuple(FileName));

            dict.SetDictTuple("segmentation_file_name",
                new HTuple(SegmentationFileName));

            return dict;
        }

    }

}
