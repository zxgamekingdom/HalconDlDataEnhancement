using System.Collections.Generic;
using System.Linq;
using HalconDotNet;
using Halcon深度学习数据增强.Abstracts;
using Halcon深度学习数据增强.Dicts.Abstracts;
using Halcon深度学习数据增强.Dicts.Extensions;
using Halcon深度学习数据增强.Dicts.Tools;

namespace Halcon深度学习数据增强.Dicts;

public class HalconClassificationDict : IHalconDict<HalconClassificationDict.Sample>
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
                LabelId = d.FromKeyTuple("image_label_id", x => x.L)
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

    public class Sample : IImageInfo, IHalconErrors
    {
        public long? LabelId { get; set; }

        public IEnumerable<string> Errors()
        {
            var errors = new List<string>();
            if (Id == null) errors.Add($"{nameof(Id)}为空");
            if (FileName.IsNullOrWhiteSpace()) errors.Add($"{nameof(FileName)}为空");
            if (LabelId == null) errors.Add($"{nameof(LabelId)}为空");

            return errors;
        }

        public string? FileName { get; set; }

        public long? Id { get; set; }

        public HDict ToHDict()
        {
            var dict = new HDict();
            dict.SetDictTuple("image_id", new HTuple(Id));
            dict.SetDictTuple("image_file_name", new HTuple(FileName));
            dict.SetDictTuple("image_label_id", new HTuple(LabelId));

            return dict;
        }
    }
}
