using System;
using System.Threading;
using System.Threading.Tasks;
using HalconDotNet;
using Halcon深度学习数据增强.Dicts;
using Halcon深度学习数据增强.Extensions;
using Halcon深度学习数据增强.Tools;
using Newtonsoft.Json;

namespace Halcon深度学习数据增强;

public static class Program
{

    [STAThread]
    public static async Task Main(string[] args)
    {
        HalconEnvironmentInit.Init();

        var s =
            @"C:\Users\Zhou Taurus\Desktop\新建文件夹 (2)\Example_ObjDetection_PillBags.hdict";

        var dict =
            HalconObjectDetectionDict.FromHDict(
                new HDict(s, new HTuple(), new HTuple()));

        JsonConvert.SerializeObject(dict, Formatting.Indented).WriteLine();
    }

}
