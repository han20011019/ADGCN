# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================

#foreach($f in gci 1_Pcap *.pcap)
#{
#    0_Tool\SplitCap_2-1\SplitCap -p 50000 -b 50000 -r $f.FullName -o 2_Session\AllLayers\$($f.BaseName)-ALL
#    # 0_Tool\SplitCap_2-1\SplitCap -p 50000 -b 50000 -r $f.FullName -s flow -o 2_Session\AllLayers\$($f.BaseName)-ALL
#    gci 2_Session\AllLayers\$($f.BaseName)-ALL | ?{$_.Length -eq 0} | del
#
#    0_Tool\SplitCap_2-1\SplitCap -p 50000 -b 50000 -r $f.FullName -o 2_Session\L7\$($f.BaseName)-L7 -y L7
#    # 0_Tool\SplitCap_2-1\SplitCap -p 50000 -b 50000 -r $f.FullName -s flow -o 2_Session\L7\$($f.BaseName)-L7 -y L7
#    gci 2_Session\L7\$($f.BaseName)-L7 | ?{$_.Length -eq 0} | del
#}
foreach($f in gci 1_Pcap *.pcap)
{
    $allLayersOutput = "2_Session\AllLayers\$($f.BaseName)-ALL"
    $l7Output = "2_Session\L7\$($f.BaseName)-L7"

    # 检查AllLayers的输出文件是否存在，如果不存在则执行SplitCap
    if (-not (Test-Path $allLayersOutput)) {
        0_Tool\SplitCap_2-1\SplitCap -p 50000 -b 50000 -r $f.FullName -o $allLayersOutput
        # 删除大小为0的文件
        gci $allLayersOutput | ?{$_.Length -eq 0} | del
    }

    # 检查L7的输出文件是否存在，如果不存在则执行SplitCap
    if (-not (Test-Path $l7Output)) {
        0_Tool\SplitCap_2-1\SplitCap -p 50000 -b 50000 -r $f.FullName -o $l7Output -y L7
        # 删除大小为0的文件
        gci $l7Output | ?{$_.Length -eq 0} | del
    }
}

0_Tool\finddupe -del 2_Session\AllLayers
0_Tool\finddupe -del 2_Session\L7