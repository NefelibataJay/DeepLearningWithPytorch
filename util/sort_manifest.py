import wave
import os


def sort_manifest(manifest, dataset_path, sorted_manifest_path):
    manifest_datas = []
    with open(manifest, 'r', encoding='utf-8') as tsv_file:
        for line in tsv_file.readlines():
            audio_path, text, speaker = line.strip().split('\t')
            manifest_datas.append((audio_path, text, speaker))

    audio_frame_lengths = []
    for audio_path, _, _ in manifest_datas:
        frame_length = get_frame_length(os.path.join(dataset_path, audio_path))
        audio_frame_lengths.append((audio_path, frame_length))

    sorted_audio_frame_lengths = sorted(audio_frame_lengths, key=lambda x: x[1])

    with open(sorted_manifest_path, 'w', encoding='utf-8') as tsv_file:
        for sorted_audio_path, frame_length in sorted_audio_frame_lengths:
            for audio_path, text, speaker in manifest_datas:
                if audio_path == sorted_audio_path:
                    tsv_file.write(f'{audio_path}\t{text}\t{speaker}\t{frame_length}\n')


def get_frame_length(wav_path):
    with wave.open(wav_path, 'rb') as wav_file:
        num_frames = wav_file.getparams().nframes
    return num_frames


if __name__ == '__main__':
    # manifests = 'D:/models/rnn-t/manifests/aishell_chars/train.tsv'
    # dataset_path = 'E:/datasets/data_aishell'
    # manifests_sorted = 'D:/models/rnn-t/manifests/aishell_chars/train.tsv'
    # sort_manifest(manifests, dataset_path, manifests_sorted)
    print('ok')
