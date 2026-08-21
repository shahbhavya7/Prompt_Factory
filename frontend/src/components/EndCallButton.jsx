export default function EndCallButton({ disabled, onClick }) {
  return (
    <button className="btn btn-primary" disabled={disabled} onClick={onClick}>
      End call &amp; learn
    </button>
  );
}
