$RemoteUserName='rindalp'
$RemoteHostName='eve.eecs.oregonstate.edu'
$PrivateKey='C:\tools\privateKey.ppk'
$SolutionDir=$PWD
$RemoteWorkingDir='/scratch/repo/cs534'

# only files with these extensions will be copied
$FileMasks='**.cpp;**.c;**.h;makefile,*.bin,*.S;*.csv;*.sh,*CMake*;*/Tools/*.txt;**.mak;thirdparty/linux/**.get'

# everything in these folders will be skipped
$ExcludeDirs='.git/;thirdparty/;Debug/;Release/;x64/;ipch/;.vs/'

C:\tools\WinSCP.com  /command `
    "open $RemoteUserName@$RemoteHostName -privatekey=""$PrivateKey"""`
    "call mkdir -p $RemoteWorkingDir"`
    "synchronize Remote $SolutionDir $RemoteWorkingDir -filemask=""$FileMasks|$ExcludeDirs;"" -transfer=binary"`
    "exit" 