<#
Windows toast for this project, kept apart from the Daily_Task toasts.

Windows groups toasts by the AppUserModelID that sends them, so sending through the PowerShell AppId (as the
Daily_Task script does) lands every project in one group. This script registers its own AppId under
HKCU\Software\Classes\AppUserModelId on first use, which is enough for an unpackaged sender to show toasts.

-Category picks the header the toast is filed under in the notification centre, and its look.
  action  the user has to do something. Stays on screen until dismissed.
  alert   something failed or stalled. Urgent sound.
  result  a run finished or a score arrived. Normal toast.

Keep it UTF-8 WITH BOM. Windows PowerShell 5.1 reads a BOM-less .ps1 as the ANSI code page and garbles the
Chinese literals below. Keep double quotes and colons out of -Message, because they break the calling command.

Usage, from the Bash tool
  powershell.exe -NoProfile -File "D:\dont_move\git_save\RSNA_Knee_Abnormality_Detection\tools\notify.ps1" -Message "<text>" -Category action
#>
param(
    [Parameter(Mandatory = $true)][string]$Message,
    [ValidateSet("action", "alert", "result")][string]$Category = "result"
)

$ErrorActionPreference = "Stop"

$appId = "ClaudeCode.RSNAKnee"
$key = "HKCU:\Software\Classes\AppUserModelId\$appId"
if (-not (Test-Path $key)) {
    New-Item -Path $key -Force | Out-Null
    New-ItemProperty -Path $key -Name DisplayName -Value "RSNA Knee" -PropertyType String -Force | Out-Null
    New-ItemProperty -Path $key -Name ShowInSettings -Value 1 -PropertyType DWord -Force | Out-Null
}

$styles = @{
    action = @{ Header = "需要你操作"; Prefix = "請處理"; Scenario = ' scenario="reminder"'; Audio = "ms-winsoundevent:Notification.Reminder" }
    alert  = @{ Header = "警告"; Prefix = "出問題了"; Scenario = ' scenario="urgent"'; Audio = "ms-winsoundevent:Notification.IM" }
    result = @{ Header = "結果"; Prefix = "完成"; Scenario = ""; Audio = "ms-winsoundevent:Notification.Default" }
}
$s = $styles[$Category]

[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null

# Message text is model-generated and may contain & or angle brackets.
function Convert-ToXmlText([string]$t) {
    $t.Replace('&', '&amp;').Replace('<', '&lt;').Replace('>', '&gt;')
}

$xml = @"
<toast$($s.Scenario)>
  <header id="rsna-$Category" title="RSNA Knee $($s.Header)" arguments="rsna-$Category" />
  <visual>
    <binding template="ToastGeneric">
      <text>$($s.Prefix)</text>
      <text>$(Convert-ToXmlText $Message)</text>
      <text placement="attribution">RSNA Knee</text>
    </binding>
  </visual>
  <audio src="$($s.Audio)" />
</toast>
"@

$doc = New-Object Windows.Data.Xml.Dom.XmlDocument
$doc.LoadXml($xml)
$toast = New-Object Windows.UI.Notifications.ToastNotification $doc
$toast.Group = "rsna-$Category"
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show($toast)

Write-Output "Toast sent ($Category)."
